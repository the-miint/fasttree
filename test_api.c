/*
 * test_api.c — Smoke tests for the FastTree library API.
 */

#include "fasttree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Small nucleotide alignment (4 sequences, 20 positions) */
static const char *nt_names[] = {"SeqA", "SeqB", "SeqC", "SeqD"};
static const char *nt_seqs[]  = {
  "ACGTACGTACGTACGTACGT",
  "ACGTACGTACGTACGTACGA",   /* 1 difference from A */
  "TGCATGCATGCATGCATGCA",   /* very different */
  "TGCATGCATGCATGCATGCG"    /* 1 difference from C */
};

/* Small protein alignment (4 sequences, 10 positions) */
static const char *aa_names[] = {"ProtA", "ProtB", "ProtC", "ProtD"};
static const char *aa_seqs[]  = {
  "ARNDCQEGHI",
  "ARNDCQEGHL",   /* 1 diff from A */
  "LKMFPSTWYV",   /* very different */
  "LKMFPSTWYA"    /* 1 diff from C */
};

static void test_basic_nucleotide(void) {
  printf("test_basic_nucleotide... ");

  fasttree_config_t config;
  fasttree_config_init(&config);
  config.seq_type = FASTTREE_SEQ_NUCLEOTIDE;
  config.seed = 12345;

  fasttree_ctx_t *ctx = fasttree_create(&config);
  assert(ctx != NULL);

  fasttree_tree_t *tree = NULL;
  fasttree_stats_t stats;
  int rc = fasttree_build(ctx, nt_names, nt_seqs, 4, 20, &tree, &stats);

  if (rc != FASTTREE_OK) {
    printf("FAIL: fasttree_build returned %d: %s\n", rc, fasttree_last_error(ctx));
    fasttree_destroy(ctx);
    return;
  }

  assert(tree != NULL);
  assert(tree->n_leaves == 4);
  assert(tree->n_nodes >= 4);

  char *newick = fasttree_tree_to_newick(tree, /*show_support*/1);
  assert(newick != NULL);
  printf("OK (n_nodes=%d, newick=%s", tree->n_nodes, newick);

  free(newick);
  fasttree_tree_free(tree);
  fasttree_destroy(ctx);
  printf(")\n");
}

static void test_basic_protein(void) {
  printf("test_basic_protein... ");

  fasttree_config_t config;
  fasttree_config_init(&config);
  config.seq_type = FASTTREE_SEQ_PROTEIN;
  config.seed = 12345;

  fasttree_ctx_t *ctx = fasttree_create(&config);
  assert(ctx != NULL);

  fasttree_tree_t *tree = NULL;
  int rc = fasttree_build(ctx, aa_names, aa_seqs, 4, 10, &tree, NULL);

  if (rc != FASTTREE_OK) {
    printf("FAIL: fasttree_build returned %d: %s\n", rc, fasttree_last_error(ctx));
    fasttree_destroy(ctx);
    return;
  }

  assert(tree != NULL);
  assert(tree->n_leaves == 4);

  char *newick = fasttree_tree_to_newick(tree, 1);
  assert(newick != NULL);
  printf("OK (newick=%s", newick);

  free(newick);
  fasttree_tree_free(tree);
  fasttree_destroy(ctx);
  printf(")\n");
}

static void test_error_handling(void) {
  printf("test_error_handling... ");

  fasttree_config_t config;
  fasttree_config_init(&config);
  config.seq_type = FASTTREE_SEQ_NUCLEOTIDE;

  fasttree_ctx_t *ctx = fasttree_create(&config);
  assert(ctx != NULL);

  /* Test with NULL sequences */
  fasttree_tree_t *tree = NULL;
  int rc = fasttree_build(ctx, NULL, NULL, 0, 0, &tree, NULL);
  assert(rc != FASTTREE_OK);

  /* Test strerror */
  assert(strcmp(fasttree_strerror(FASTTREE_OK), "Success") == 0);
  assert(strcmp(fasttree_strerror(FASTTREE_ERR_NOMEM), "Out of memory") == 0);
  assert(strcmp(fasttree_strerror(-999), "Unknown error") == 0);

  fasttree_destroy(ctx);
  printf("OK\n");
}

int main(void) {
  printf("FastTree API smoke tests\n");
  printf("========================\n");

  test_basic_nucleotide();
  test_basic_protein();
  test_error_handling();

  printf("All tests passed!\n");
  return 0;
}
