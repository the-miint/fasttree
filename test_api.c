/*
 * test_api.c — Tests for the FastTree library API.
 */

#include "fasttree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int n_tests = 0;
static int n_passed = 0;

#define TEST_START(name) do { printf("  %-45s ", name); n_tests++; } while(0)
#define TEST_PASS()      do { n_passed++; printf("OK\n"); return; } while(0)
#define TEST_FAIL(msg)   do { printf("FAIL: %s\n", msg); return; } while(0)

/* ── Test data ──────────────────────────────────────────────────── */

static const char *nt_names[] = {"SeqA", "SeqB", "SeqC", "SeqD"};
static const char *nt_seqs[]  = {
  "ACGTACGTACGTACGTACGT",
  "ACGTACGTACGTACGTACGA",
  "TGCATGCATGCATGCATGCA",
  "TGCATGCATGCATGCATGCG"
};

static const char *aa_names[] = {"ProtA", "ProtB", "ProtC", "ProtD"};
static const char *aa_seqs[]  = {
  "ARNDCQEGHI",
  "ARNDCQEGHL",
  "LKMFPSTWYV",
  "LKMFPSTWYA"
};

/* Alignment with duplicate sequences */
static const char *dup_names[] = {"Uniq1", "Dup1a", "Dup1b", "Uniq2"};
static const char *dup_seqs[]  = {
  "ACGTACGTACGTACGTACGT",
  "TGCATGCATGCATGCATGCA",  /* identical to Dup1b */
  "TGCATGCATGCATGCATGCA",  /* identical to Dup1a */
  "AAAAAAAAAAAAAAAAAAGT"
};

/* ── Helper ──────────────────────────────────────────────────────── */

static fasttree_ctx_t *make_ctx(fasttree_seqtype_t st) {
  fasttree_config_t config;
  fasttree_config_init(&config);
  config.seq_type = st;
  config.seed = 12345;
  return fasttree_create(&config);
}

/* ── Tests ───────────────────────────────────────────────────────── */

static void test_nucleotide_aos(void) {
  TEST_START("nucleotide AOS");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  fasttree_tree_t *tree = NULL;
  fasttree_stats_t stats;
  int rc = fasttree_build(ctx, nt_names, nt_seqs, 4, 20, &tree, &stats);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));

  assert(tree && tree->n_leaves == 4 && tree->n_nodes >= 4);
  assert(stats.n_unique_seqs == 4);

  char *nwk = fasttree_tree_to_newick(tree, 1);
  assert(nwk && strlen(nwk) > 10);

  free(nwk);
  fasttree_tree_free(tree);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_protein_aos(void) {
  TEST_START("protein AOS");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_PROTEIN);
  assert(ctx);

  fasttree_tree_t *tree = NULL;
  int rc = fasttree_build(ctx, aa_names, aa_seqs, 4, 10, &tree, NULL);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));

  assert(tree && tree->n_leaves == 4);
  char *nwk = fasttree_tree_to_newick(tree, 1);
  assert(nwk);

  free(nwk);
  fasttree_tree_free(tree);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_nucleotide_soa(void) {
  TEST_START("nucleotide SOA");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  fasttree_tree_soa_t *tree = NULL;
  fasttree_stats_t stats;
  int rc = fasttree_build_soa(ctx, nt_names, nt_seqs, 4, 20, &tree, &stats);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));

  assert(tree && tree->n_leaves == 4 && tree->n_nodes >= 4);
  assert(tree->parent && tree->branch_length && tree->is_leaf);

  int leaf_count = 0, i;
  for (i = 0; i < tree->n_nodes; i++)
    if (tree->is_leaf[i]) leaf_count++;
  assert(leaf_count == tree->n_leaves);

  char *nwk = fasttree_tree_soa_to_newick(tree, 1);
  assert(nwk && strlen(nwk) > 10);

  free(nwk);
  fasttree_tree_soa_free(tree);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_soa_newick_matches_aos(void) {
  TEST_START("SOA Newick matches AOS (same context)");
  /* Use a single context to isolate layout differences only.
     Both builds use the same seed/RNG so output must be identical. */
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  fasttree_tree_t *aos = NULL;
  int rc1 = fasttree_build(ctx, nt_names, nt_seqs, 4, 20, &aos, NULL);
  assert(rc1 == FASTTREE_OK);

  fasttree_tree_soa_t *soa = NULL;
  int rc2 = fasttree_build_soa(ctx, nt_names, nt_seqs, 4, 20, &soa, NULL);
  assert(rc2 == FASTTREE_OK);

  char *nwk_aos = fasttree_tree_to_newick(aos, 1);
  char *nwk_soa = fasttree_tree_soa_to_newick(soa, 1);
  assert(nwk_aos && nwk_soa);

  if (strcmp(nwk_aos, nwk_soa) != 0) {
    char msg[256];
    snprintf(msg, sizeof(msg), "AOS != SOA Newick");
    free(nwk_aos); free(nwk_soa);
    fasttree_tree_free(aos); fasttree_tree_soa_free(soa);
    fasttree_destroy(ctx);
    TEST_FAIL(msg);
  }

  free(nwk_aos); free(nwk_soa);
  fasttree_tree_free(aos); fasttree_tree_soa_free(soa);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_duplicate_sequences_aos(void) {
  TEST_START("duplicate sequences (AOS)");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  fasttree_tree_t *tree = NULL;
  fasttree_stats_t stats;
  int rc = fasttree_build(ctx, dup_names, dup_seqs, 4, 20, &tree, &stats);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));

  assert(tree->n_leaves == 4);
  assert(stats.n_unique_seqs == 3);

  char *nwk = fasttree_tree_to_newick(tree, 0);
  assert(nwk);
  assert(strstr(nwk, "Uniq1") && strstr(nwk, "Dup1a") &&
         strstr(nwk, "Dup1b") && strstr(nwk, "Uniq2"));

  free(nwk);
  fasttree_tree_free(tree);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_duplicate_sequences_soa(void) {
  TEST_START("duplicate sequences (SOA)");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  fasttree_tree_soa_t *tree = NULL;
  fasttree_stats_t stats;
  int rc = fasttree_build_soa(ctx, dup_names, dup_seqs, 4, 20, &tree, &stats);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));

  assert(tree->n_leaves == 4);
  assert(stats.n_unique_seqs == 3);

  char *nwk = fasttree_tree_soa_to_newick(tree, 0);
  assert(nwk);
  assert(strstr(nwk, "Uniq1") && strstr(nwk, "Dup1a") &&
         strstr(nwk, "Dup1b") && strstr(nwk, "Uniq2"));

  free(nwk);
  fasttree_tree_soa_free(tree);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_context_reuse(void) {
  TEST_START("context reuse (two builds)");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  fasttree_tree_t *tree1 = NULL;
  int rc = fasttree_build(ctx, nt_names, nt_seqs, 4, 20, &tree1, NULL);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));
  char *nwk1 = fasttree_tree_to_newick(tree1, 1);

  fasttree_tree_t *tree2 = NULL;
  rc = fasttree_build(ctx, nt_names, nt_seqs, 4, 20, &tree2, NULL);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));
  char *nwk2 = fasttree_tree_to_newick(tree2, 1);

  if (strcmp(nwk1, nwk2) != 0) {
    free(nwk1); free(nwk2);
    fasttree_tree_free(tree1); fasttree_tree_free(tree2);
    fasttree_destroy(ctx);
    TEST_FAIL("builds not identical");
  }

  free(nwk1); free(nwk2);
  fasttree_tree_free(tree1); fasttree_tree_free(tree2);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_error_recovery(void) {
  TEST_START("error recovery (bad input then good)");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  /* Trigger error: wrong sequence length */
  const char *bad_names[] = {"A", "B"};
  const char *bad_seqs[]  = {"ACGT", "AC"};
  fasttree_tree_t *tree = NULL;
  int rc = fasttree_build(ctx, bad_names, bad_seqs, 2, 4, &tree, NULL);
  assert(rc == FASTTREE_ERR_INVALID_INPUT);
  assert(tree == NULL);
  assert(strlen(fasttree_last_error(ctx)) > 0);

  /* Good build on same context should succeed */
  rc = fasttree_build(ctx, nt_names, nt_seqs, 4, 20, &tree, NULL);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));
  assert(tree && tree->n_leaves == 4);

  fasttree_tree_free(tree);
  fasttree_destroy(ctx);
  TEST_PASS();
}

static void test_error_handling(void) {
  TEST_START("error codes and strerror");

  assert(strcmp(fasttree_strerror(FASTTREE_OK), "Success") == 0);
  assert(strcmp(fasttree_strerror(FASTTREE_ERR_NOMEM), "Out of memory") == 0);
  assert(strcmp(fasttree_strerror(FASTTREE_ERR_INVALID_INPUT), "Invalid input") == 0);
  assert(strcmp(fasttree_strerror(-999), "Unknown error") == 0);

  /* NULL context returns error */
  int rc = fasttree_build(NULL, NULL, NULL, 0, 0, NULL, NULL);
  assert(rc == FASTTREE_ERR_INVALID_INPUT);

  /* start_newick rejected */
  fasttree_config_t config;
  fasttree_config_init(&config);
  config.start_newick = "(A,B);";
  fasttree_ctx_t *ctx = fasttree_create(&config);
  assert(ctx == NULL);

  TEST_PASS();
}

static void test_soa_columnar_access(void) {
  TEST_START("SOA columnar access + children_offset");
  fasttree_ctx_t *ctx = make_ctx(FASTTREE_SEQ_NUCLEOTIDE);
  assert(ctx);

  fasttree_tree_soa_t *tree = NULL;
  int rc = fasttree_build_soa(ctx, nt_names, nt_seqs, 4, 20, &tree, NULL);
  if (rc != FASTTREE_OK) TEST_FAIL(fasttree_last_error(ctx));

  /* Sum branch lengths (columnar) */
  double total = 0;
  int i;
  for (i = 0; i < tree->n_nodes; i++)
    total += tree->branch_length[i];
  assert(total > 0);

  /* Count internal nodes */
  int internal = 0;
  for (i = 0; i < tree->n_nodes; i++)
    if (!tree->is_leaf[i]) internal++;
  assert(internal + tree->n_leaves == tree->n_nodes);

  /* Root has no parent */
  assert(tree->parent[tree->root] == -1);

  /* Leaf nodes have children_offset == -1 (poisoned) */
  for (i = 0; i < tree->n_nodes; i++) {
    if (tree->is_leaf[i])
      assert(tree->children_offset[i] == -1);
  }

  /* Verify children_offset + parent consistency for internal nodes */
  for (i = 0; i < tree->n_nodes; i++) {
    if (!tree->is_leaf[i]) {
      int off = tree->children_offset[i];
      int nc  = tree->n_children[i];
      assert(off >= 0 && nc > 0);
      int j;
      for (j = 0; j < nc; j++) {
        int child = tree->_children_buf[off + j];
        assert(child >= 0 && child < tree->n_nodes);
        assert(tree->parent[child] == i);
      }
    }
  }

  fasttree_tree_soa_free(tree);
  fasttree_destroy(ctx);
  TEST_PASS();
}

int main(void) {
  printf("FastTree API tests\n");
  printf("==================\n");

  test_nucleotide_aos();
  test_protein_aos();
  test_nucleotide_soa();
  test_soa_newick_matches_aos();
  test_soa_columnar_access();
  test_duplicate_sequences_aos();
  test_duplicate_sequences_soa();
  test_context_reuse();
  test_error_recovery();
  test_error_handling();

  printf("------------------\n");
  printf("%d/%d tests passed\n", n_passed, n_tests);
  return n_passed == n_tests ? 0 : 1;
}
