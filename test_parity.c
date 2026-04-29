/* test_parity.c — Run library on a PHYLIP file, write Newick.
   Used to compare library output bit-for-bit against FastTree.orig. */

#include "fasttree.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Parse interleaved PHYLIP (matches what FastTree's ReadAlignment does for PHYLIP). */
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

    /* Trim trailing CR */
    while (end > line && (end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t')) end--;
    if (line == end) {
      if (iSeq == n || iSeq == 0) iSeq = 0;
      continue;
    }

    int j = 0;
    char *q = line;
    if (*q == ' ') {
      /* sequence-only line for the current iSeq */
    } else {
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

  for (i = 0; i < n; i++)
    if ((int)strlen(seqs[i]) != L) return -4;

  *n_out = n;
  *L_out = L;
  *names_out = names;
  *seqs_out = seqs;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <phylip> [-nt]\n", argv[0]); return 2; }
  int is_nt = 0;
  int i;
  for (i = 2; i < argc; i++) if (strcmp(argv[i], "-nt") == 0) is_nt = 1;

  FILE *fp = fopen(argv[1], "r");
  if (!fp) { perror(argv[1]); return 2; }
  size_t flen;
  char *text = read_all(fp, &flen);
  fclose(fp);

  int n, L;
  char **names = NULL;
  char **seqs = NULL;
  int rc = parse_phylip(text, &n, &L, &names, &seqs);
  if (rc != 0) { fprintf(stderr, "parse_phylip failed rc=%d\n", rc); return 3; }

  fprintf(stderr, "Parsed %d sequences x %d positions\n", n, L);

  fasttree_config_t cfg;
  fasttree_config_init(&cfg);
  cfg.seq_type = is_nt ? FASTTREE_SEQ_NUCLEOTIDE : FASTTREE_SEQ_PROTEIN;
  cfg.seed = 12345;

  fasttree_ctx_t *ctx = fasttree_create(&cfg);
  if (!ctx) { fprintf(stderr, "fasttree_create failed\n"); return 4; }

  fasttree_tree_soa_t *tree = NULL;
  fasttree_stats_t stats;
  rc = fasttree_build_soa(ctx, (const char**)names, (const char**)seqs, n, L, &tree, &stats);
  if (rc != FASTTREE_OK) {
    fprintf(stderr, "build failed rc=%d: %s\n", rc, fasttree_last_error(ctx));
    return 5;
  }

  char *nwk = fasttree_tree_soa_to_newick(tree, 1);
  fputs(nwk, stdout);
  free(nwk);
  fasttree_tree_soa_free(tree);
  fasttree_destroy(ctx);
  return 0;
}
