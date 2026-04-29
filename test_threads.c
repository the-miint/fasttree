/* test_threads.c — thread-safety reproducer for fasttree_build_soa.
   Spawns N threads, each builds its own ctx + tree on the same input.
   Each thread's Newick must match the supplied ground-truth file. */

#include "fasttree.h"
#include <ctype.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS 32

static char *read_all(FILE *fp, size_t *len_out) {
  size_t cap = 1 << 16, len = 0;
  char *buf = malloc(cap);
  for (;;) {
    if (len + 4096 > cap) { cap *= 2; buf = realloc(buf, cap); }
    size_t n = fread(buf + len, 1, cap - len, fp);
    if (n == 0) break;
    len += n;
  }
  buf[len] = '\0';
  *len_out = len;
  return buf;
}

/* Same PHYLIP parser as test_parity.c. */
static int parse_phylip(char *text, int *n_out, int *L_out, char ***names_out, char ***seqs_out) {
  char *p = text;
  while (*p == ' ' || *p == '\t') p++;
  int n = 0, L = 0;
  if (sscanf(p, "%d %d", &n, &L) != 2 || n < 1 || L < 1) return -1;
  while (*p && *p != '\n') p++;
  if (*p) p++;

  char **names = calloc(n, sizeof(char*));
  char **seqs  = calloc(n, sizeof(char*));
  int i;
  for (i = 0; i < n; i++) seqs[i] = calloc(L + 1, 1);

  int iSeq = 0;
  while (*p) {
    char *line = p;
    while (*p && *p != '\n') p++;
    char *end = p;
    if (*p == '\n') p++;
    while (end > line && (end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t')) end--;
    if (line == end) {
      if (iSeq == n || iSeq == 0) iSeq = 0;
      continue;
    }
    int j = 0;
    char *q = line;
    if (*q != ' ') {
      while (q + j < end && q[j] != ' ' && q[j] != '\t') j++;
      if (q + j >= end || j == 0) return -2;
      if (names[iSeq] == NULL) {
        names[iSeq] = malloc(j + 1);
        memcpy(names[iSeq], q, j);
        names[iSeq][j] = '\0';
      }
    }
    int seqlen = strlen(seqs[iSeq]);
    int k;
    for (k = j; q + k < end; k++) {
      char c = q[k];
      if (c == ' ' || c == '\t') continue;
      if (seqlen >= L) return -3;
      seqs[iSeq][seqlen++] = (char)toupper((unsigned char)c);
    }
    seqs[iSeq][seqlen] = '\0';
    iSeq++;
    if (iSeq == n && (int)strlen(seqs[0]) == L) break;
  }
  for (i = 0; i < n; i++) if ((int)strlen(seqs[i]) != L) return -4;

  *n_out = n; *L_out = L; *names_out = names; *seqs_out = seqs;
  return 0;
}

typedef struct {
  int tid;
  int is_nt;
  int n;
  int L;
  const char **names;
  const char **seqs;
  const char *expected_newick;
  int ok;
  char err[256];
} thread_args_t;

static void *thread_main(void *arg) {
  thread_args_t *ta = (thread_args_t *)arg;
  fasttree_config_t cfg;
  fasttree_config_init(&cfg);
  cfg.seq_type = ta->is_nt ? FASTTREE_SEQ_NUCLEOTIDE : FASTTREE_SEQ_PROTEIN;
  cfg.seed = 12345;

  fasttree_ctx_t *ctx = fasttree_create(&cfg);
  if (!ctx) { snprintf(ta->err, sizeof(ta->err), "fasttree_create"); return NULL; }

  fasttree_tree_soa_t *tree = NULL;
  int rc = fasttree_build_soa(ctx, ta->names, ta->seqs, ta->n, ta->L, &tree, NULL);
  if (rc != FASTTREE_OK) {
    snprintf(ta->err, sizeof(ta->err), "build rc=%d: %s", rc, fasttree_last_error(ctx));
    fasttree_destroy(ctx);
    return NULL;
  }

  char *nwk = fasttree_tree_soa_to_newick(tree, 1);
  if (!nwk) { snprintf(ta->err, sizeof(ta->err), "newick alloc"); fasttree_tree_soa_free(tree); fasttree_destroy(ctx); return NULL; }

  if (strcmp(nwk, ta->expected_newick) != 0) {
    snprintf(ta->err, sizeof(ta->err), "newick mismatch (len got=%zu, expected=%zu)",
             strlen(nwk), strlen(ta->expected_newick));
  } else {
    ta->ok = 1;
  }
  free(nwk);
  fasttree_tree_soa_free(tree);
  fasttree_destroy(ctx);
  return NULL;
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <phylip> <expected.nwk> <nthreads> [-nt]\n", argv[0]); return 2; }
  int is_nt = 0; int i;
  for (i = 4; i < argc; i++) if (strcmp(argv[i], "-nt") == 0) is_nt = 1;
  int nthreads = atoi(argv[3]);
  if (nthreads < 1 || nthreads > MAX_THREADS) { fprintf(stderr, "bad nthreads\n"); return 2; }

  FILE *fp = fopen(argv[1], "r");
  if (!fp) { perror(argv[1]); return 2; }
  size_t flen; char *text = read_all(fp, &flen); fclose(fp);
  int n, L; char **names = NULL; char **seqs = NULL;
  int rc = parse_phylip(text, &n, &L, &names, &seqs);
  if (rc != 0) { fprintf(stderr, "parse rc=%d\n", rc); return 3; }

  fp = fopen(argv[2], "r");
  if (!fp) { perror(argv[2]); return 2; }
  size_t elen; char *expected = read_all(fp, &elen); fclose(fp);

  pthread_t threads[MAX_THREADS];
  thread_args_t args[MAX_THREADS];
  for (i = 0; i < nthreads; i++) {
    args[i].tid = i;
    args[i].is_nt = is_nt;
    args[i].n = n; args[i].L = L;
    args[i].names = (const char**)names;
    args[i].seqs  = (const char**)seqs;
    args[i].expected_newick = expected;
    args[i].ok = 0;
    args[i].err[0] = '\0';
    pthread_create(&threads[i], NULL, thread_main, &args[i]);
  }

  int n_ok = 0;
  for (i = 0; i < nthreads; i++) {
    pthread_join(threads[i], NULL);
    if (args[i].ok) {
      n_ok++;
    } else {
      fprintf(stderr, "thread %d FAIL: %s\n", i, args[i].err);
    }
  }
  fprintf(stderr, "%d/%d threads OK\n", n_ok, nthreads);
  return n_ok == nthreads ? 0 : 1;
}
