/*
 * fasttree_api.c — Public API implementation for the FastTree library.
 *
 * Thin wrappers that create a context, call core functions, and
 * extract results.  The heavy lifting is in fasttree_core.c.
 */

#include "fasttree_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* ── Config defaults ──────────────────────────────────────────────── */

void fasttree_config_init(fasttree_config_t *config) {
  memset(config, 0, sizeof(*config));
  config->struct_size = sizeof(fasttree_config_t);

  config->seq_type = FASTTREE_SEQ_AUTO;
  config->model    = FASTTREE_MODEL_AUTO;

  /* GTR */
  int i;
  for (i = 0; i < 6; i++) config->gtr_rates[i] = 1.0;
  for (i = 0; i < 4; i++) config->gtr_freq[i] = 0.25;
  config->gtr_from_alignment = 1;

  /* Tree search */
  config->nni_rounds    = -1;    /* auto */
  config->spr_rounds    = 2;
  config->ml_nni_rounds = -1;    /* auto */
  config->n_rate_cats   = 20;
  config->slow          = 0;
  config->fastest       = 0;

  /* Support */
  config->n_bootstrap    = 1000;
  config->gamma_log_lk   = 0;

  /* Reproducibility */
  config->seed = 314159;

  /* Heuristics */
  config->use_top_hits       = 1;
  config->top_hits_mult      = 1.0;
  config->use_bionj          = 0;
  config->pseudo_weight      = 0.0;
  config->constraint_weight  = 100.0;
  config->ml_accuracy        = 1;

  config->start_newick  = NULL;
  config->quote_names   = 0;
  config->n_threads     = 0;

  config->progress_callback  = NULL;
  config->progress_user_data = NULL;
  config->log_callback       = NULL;
  config->log_user_data      = NULL;

  config->alloc_fn        = NULL;  /* NULL = use system malloc/free */
  config->free_fn         = NULL;
  config->alloc_user_data = NULL;
}

/* ── Context lifecycle ────────────────────────────────────────────── */

fasttree_ctx_t *fasttree_create(const fasttree_config_t *config) {
  if (config == NULL || config->struct_size < sizeof(fasttree_config_t))
    return NULL;

  /* Reject unimplemented options that would silently produce wrong results */
  if (config->start_newick != NULL)
    return NULL;  /* start_newick not yet supported; use CLI -intree instead */

  fasttree_ctx_t *ctx = (fasttree_ctx_t *)malloc(sizeof(fasttree_ctx_t));
  if (ctx == NULL) return NULL;

  fasttree_ctx_init(ctx);

  /* Sequence type → nCodes, useMatrix, logdist */
  switch (config->seq_type) {
    case FASTTREE_SEQ_NUCLEOTIDE:
      ctx->nCodes = 4;
      ctx->useMatrix = false;
      ctx->logdist = true;
      break;
    case FASTTREE_SEQ_PROTEIN:
      ctx->nCodes = 20;
      ctx->useMatrix = true;
      ctx->logdist = true;
      break;
    case FASTTREE_SEQ_AUTO:
    default:
      ctx->nCodes = 20;  /* default to protein; fasttree_build auto-detects */
      ctx->useMatrix = true;
      ctx->logdist = true;
      break;
  }

  /* Algorithm options */
  ctx->slow         = config->slow;
  ctx->fastest      = config->fastest;
  ctx->bionj        = config->use_bionj;
  ctx->pseudoWeight = config->pseudo_weight;
  ctx->constraintWeight = config->constraint_weight;
  ctx->mlAccuracy   = config->ml_accuracy;
  ctx->bQuote       = config->quote_names;
  ctx->bGammaLogLk  = config->gamma_log_lk;

  /* Top-hits: use_top_hits=0 disables by setting tophitsMult=0 */
  ctx->tophitsMult  = config->use_top_hits ? config->top_hits_mult : 0.0;

  /* Callbacks */
  ctx->log_callback       = config->log_callback;
  ctx->log_user_data      = config->log_user_data;
  ctx->progress_callback  = config->progress_callback;
  ctx->progress_user_data = config->progress_user_data;

  /* Config snapshot for fasttree_build */
  ctx->seed              = config->seed;
  ctx->nni_rounds        = config->nni_rounds;
  ctx->spr_rounds        = config->spr_rounds;
  ctx->ml_nni_rounds     = config->ml_nni_rounds;
  ctx->n_rate_cats       = config->n_rate_cats;
  ctx->n_bootstrap       = config->n_bootstrap;
  ctx->n_threads         = config->n_threads;
  ctx->model             = (int)config->model;
  ctx->gtr_from_alignment = config->gtr_from_alignment;
  int i;
  for (i = 0; i < 6; i++) ctx->gtr_rates[i] = config->gtr_rates[i];
  for (i = 0; i < 4; i++) ctx->gtr_freq[i]  = config->gtr_freq[i];

  /* Custom allocator — wire into the arena */
  if (config->alloc_fn) {
    ctx->arena.alloc_fn       = config->alloc_fn;
    ctx->arena.free_fn        = config->free_fn;
    ctx->arena.alloc_user_data = config->alloc_user_data;
  }

  return ctx;
}

void fasttree_destroy(fasttree_ctx_t *ctx) {
  if (ctx == NULL) return;
  ft_arena_destroy(&ctx->arena);
  if (ctx->start_newick) free((void *)ctx->start_newick);
  free(ctx);
}

/* ── Build ────────────────────────────────────────────────────────── */

int fasttree_build(fasttree_ctx_t *ctx,
                   const char **names, const char **seqs,
                   int nSeq, int nPos,
                   fasttree_tree_t **tree_out,
                   fasttree_stats_t *stats_out) {
  if (ctx == NULL || names == NULL || seqs == NULL || tree_out == NULL)
    return FASTTREE_ERR_INVALID_INPUT;

  fasttree_ctx_t *ft_ctx = ctx;  /* alias for FT_ macros */
  *tree_out = NULL;
  ft_ctx->error_msg[0] = '\0';  /* clear stale error */

  /* Reset arena for this build (frees any memory from previous build/error).
     Preserve custom allocator pointers across the reset. */
  {
    void *(*saved_alloc)(size_t, void*) = ft_ctx->arena.alloc_fn;
    void  (*saved_free)(void*, void*)   = ft_ctx->arena.free_fn;
    void   *saved_ud                    = ft_ctx->arena.alloc_user_data;
    ft_arena_destroy(&ft_ctx->arena);
    ft_arena_init(&ft_ctx->arena);
    ft_ctx->arena.alloc_fn        = saved_alloc;
    ft_ctx->arena.free_fn         = saved_free;
    ft_ctx->arena.alloc_user_data = saved_ud;
  }

#ifdef OPENMP
  /* Set thread count per-build, not per-create (avoids global race) */
  if (ft_ctx->n_threads > 0)
    omp_set_num_threads(ft_ctx->n_threads);
#endif

  /* Set up longjmp error recovery */
  int err = setjmp(ft_ctx->error_jmp);
  if (err != 0) {
    ft_ctx->error_code = err;
    return err;
  }

  /* Initialize RNG */
  ran_start(ft_ctx, (long)ft_ctx->seed);

  /* Set up codesString based on nCodes (set by fasttree_create from config) */
  bool isNucleotide = (FT_nCodes == 4);
  FT_codesString = isNucleotide ? FT_codesStringNT : FT_codesStringAA;

  /* Reset charToCode for the new codesString */
  FT_charToCodeSet = 0;

  /* Reset performance counters */
  FT_profileOps = 0; FT_outprofileOps = 0; FT_seqOps = 0;
  FT_profileAvgOps = 0; FT_nHillBetter = 0; FT_nCloseUsed = 0;
  FT_nClose2Used = 0; FT_nRefreshTopHits = 0; FT_nVisibleUpdate = 0;
  FT_nNNI = 0; FT_nSPR = 0; FT_nML_NNI = 0;
  FT_nProfileFreqAlloc = 0; FT_nProfileFreqAvoid = 0;
  FT_szAllAlloc = 0; FT_mymallocUsed = 0; FT_maxmallocHeap = 0;
  FT_nLkCompute = 0; FT_nPosteriorCompute = 0;
  FT_nAAPosteriorExact = 0; FT_nAAPosteriorRough = 0; FT_nStarTests = 0;

  /* Reset progress timer */
  FT_time_set = false;

  /* Copy alignment */
  alignment_t *aln = AlignmentFromMemory(ft_ctx, names, seqs, nSeq, nPos);
  ProgressReport(ft_ctx, "Read alignment", 0, 0, 0, 0);

  /* Check unique names */
  hashstrings_t *hashnames = MakeHashtable(ft_ctx, aln->names, aln->nSeq);
  int i;
  for (i = 0; i < aln->nSeq; i++) {
    hashiterator_t hi = FindMatch(ft_ctx, hashnames, aln->names[i]);
    if (HashCount(ft_ctx, hashnames, hi) != 1) {
      snprintf(ft_ctx->error_msg, sizeof(ft_ctx->error_msg),
               "Non-unique name '%s' in the alignment", aln->names[i]);
      longjmp(ft_ctx->error_jmp, FASTTREE_ERR_INVALID_INPUT);
    }
  }

  /* Deduplicate */
  uniquify_t *unique = UniquifyAln(ft_ctx, aln);
  ProgressReport(ft_ctx, "Identified unique sequences", 0, 0, 0, 0);

  /* Set up distance matrix */
  distance_matrix_t *distance_matrix = NULL;
  if (FT_useMatrix) {
    distance_matrix = &matrixBLOSUM45;
    SetupDistanceMatrix(ft_ctx, distance_matrix);
  }

  /* Set up transition matrix based on model config */
  transition_matrix_t *transmat = NULL;
  bool bUseGtr = (ft_ctx->model == FASTTREE_MODEL_GTR);
  bool resetGtr = false;
  if (FT_nCodes == 20) {
    switch (ft_ctx->model) {
      case FASTTREE_MODEL_LG:
        transmat = CreateTransitionMatrix(ft_ctx, matrixLG08, statLG08);
        break;
      case FASTTREE_MODEL_WAG:
        transmat = CreateTransitionMatrix(ft_ctx, matrixWAG01, statWAG01);
        break;
      default: /* JTT, AUTO */
        transmat = CreateTransitionMatrix(ft_ctx, matrixJTT92, statJTT92);
        break;
    }
  } else if (FT_nCodes == 4 && bUseGtr) {
    if (ft_ctx->gtr_from_alignment) {
      /* GTR auto: transmat stays NULL; SetMLGtr called after initial tree */
      resetGtr = true;
    } else {
      transmat = CreateGTR(ft_ctx, ft_ctx->gtr_rates, ft_ctx->gtr_freq);
    }
  }

  /* Initialize NJ */
  NJ_t *NJ = InitNJ(ft_ctx, unique->uniqueSeq, unique->nUnique, aln->nPos,
                     /*constraintSeqs*/NULL, /*nConstraints*/0,
                     distance_matrix, transmat);
  FreeAlignmentSeqs(ft_ctx, aln);

  /* Build initial tree */
  FastNJ(ft_ctx, NJ);

  /* NNI/SPR rounds */
  int nniToDo = ft_ctx->nni_rounds == -1
    ? (int)(0.5 + 4.0 * log(NJ->nSeq) / log(2))
    : ft_ctx->nni_rounds;
  int sprRemaining = ft_ctx->spr_rounds;
  int maxSPRLength = 10;
  int MLnniToDo = ft_ctx->ml_nni_rounds == -1
    ? (int)(0.5 + 2.0 * log(NJ->nSeq) / log(2))
    : ft_ctx->ml_nni_rounds;
  int nRateCats = ft_ctx->n_rate_cats;
  int nBootstrap = ft_ctx->n_bootstrap;

  if (nniToDo > 0) {
    bool bConverged = false;
    nni_stats_t *nni_stats = InitNNIStats(ft_ctx, NJ);
    for (i = 0; i < nniToDo; i++) {
      double maxDelta;
      if (!bConverged) {
        int nChange = NNI(ft_ctx, NJ, i, nniToDo, /*useML*/false, nni_stats, &maxDelta);
        if (nChange == 0) bConverged = true;
      }
      if (sprRemaining > 0 && (nniToDo/(ft_ctx->spr_rounds+1) > 0
          && ((i+1) % (nniToDo/(ft_ctx->spr_rounds+1))) == 0)) {
        SPR(ft_ctx, NJ, maxSPRLength, ft_ctx->spr_rounds - sprRemaining, ft_ctx->spr_rounds);
        sprRemaining--;
        bConverged = false;
        nni_stats = FreeNNIStats(ft_ctx, nni_stats, NJ);
        nni_stats = InitNNIStats(ft_ctx, NJ);
      }
    }
    nni_stats = FreeNNIStats(ft_ctx, nni_stats, NJ);
  }
  while (sprRemaining > 0) {
    SPR(ft_ctx, NJ, maxSPRLength, ft_ctx->spr_rounds - sprRemaining, ft_ctx->spr_rounds);
    sprRemaining--;
  }

  UpdateBranchLengths(ft_ctx, NJ);

  SplitCount_t splitcount = {0, 0, 0, 0, 0.0, 0.0};
  double final_loglk = -1.0;

  if (MLnniToDo > 0) {
    /* Convert profiles for ML */
    distance_matrix_t *tmatAsDist = TransMatToDistanceMat(ft_ctx, NJ->transmat);
    RecomputeProfiles(ft_ctx, NJ, tmatAsDist);
    tmatAsDist = myfree(ft_ctx, tmatAsDist, sizeof(distance_matrix_t));

    nni_stats_t *nni_stats = InitNNIStats(ft_ctx, NJ);
    double lastloglk = -1e20;
    bool bConverged = false;

    OptimizeAllBranchLengths(ft_ctx, NJ);

    if (resetGtr)
      SetMLGtr(ft_ctx, NJ, NULL, NULL);
    SetMLRates(ft_ctx, NJ, nRateCats);

    /* ML NNI rounds — matches reference in main() exactly */
    for (i = 0; i < MLnniToDo; i++) {
      double maxDelta;
      NNI(ft_ctx, NJ, i, MLnniToDo, /*useML*/true, nni_stats, &maxDelta);
      double loglk = TreeLogLk(ft_ctx, NJ, NULL);
      bool bConvergedHere = (i > 0) && ((loglk < lastloglk + FT_treeLogLkDelta) || maxDelta < FT_treeLogLkDelta);
      if (bConverged)
        break;  /* did our extra round */
      if (bConvergedHere)
        bConverged = true;
      if (bConverged || i == MLnniToDo - 2) {
        /* Final round uses high-accuracy settings — reset NNI stats */
        nni_stats = FreeNNIStats(ft_ctx, nni_stats, NJ);
        nni_stats = InitNNIStats(ft_ctx, NJ);
      }
      lastloglk = loglk;
      if (i == 0 && NJ->rates.nRateCategories == 1) {
        if (resetGtr)
          SetMLGtr(ft_ctx, NJ, NULL, NULL);
        SetMLRates(ft_ctx, NJ, nRateCats);
      }
    }
    nni_stats = FreeNNIStats(ft_ctx, nni_stats, NJ);

    OptimizeAllBranchLengths(ft_ctx, NJ);
    final_loglk = TreeLogLk(ft_ctx, NJ, NULL);

    if (!FT_fastest || nBootstrap > 0)
      TestSplitsML(ft_ctx, NJ, &splitcount, nBootstrap);

    if (ft_ctx->bGammaLogLk && nRateCats > 1) {
      numeric_t *rates = MLSiteRates(ft_ctx, nRateCats);
      double *site_loglk = MLSiteLikelihoodsByRate(ft_ctx, NJ, rates, nRateCats);
      double scale = RescaleGammaLogLk(ft_ctx, NJ->nPos, nRateCats, rates, site_loglk, NULL);
      rates = myfree(ft_ctx, rates, sizeof(numeric_t) * nRateCats);
      site_loglk = myfree(ft_ctx, site_loglk, sizeof(double) * nRateCats * NJ->nPos);
      for (i = 0; i < NJ->maxnodes; i++)
        NJ->branchlength[i] *= scale;
    }
  } else {
    TestSplitsMinEvo(ft_ctx, NJ, &splitcount);
    if (nBootstrap > 0)
      ReliabilityNJ(ft_ctx, NJ, nBootstrap);
  }

  /* ── Extract tree into fasttree_tree_t ──
     Handles duplicate sequences: unique leaves with duplicates get
     zero-branch-length leaf children, matching PrintNJ behavior. */
  {
    int n_unique = NJ->nSeq;  /* unique leaf count in NJ */
    int root = NJ->root;

    /* Count duplicate sequences */
    int n_dup_leaves = 0;
    for (i = 0; i < n_unique; i++) {
      int alnIdx = unique->alnNext[unique->uniqueFirst[i]];
      while (alnIdx >= 0) {
        n_dup_leaves++;
        alnIdx = unique->alnNext[alnIdx];
      }
    }

    int n_nodes = NJ->maxnode + n_dup_leaves;
    int n_leaves_total = n_unique + n_dup_leaves;

    fasttree_tree_t *tree = (fasttree_tree_t *)malloc(sizeof(fasttree_tree_t));
    if (!tree) { snprintf(ft_ctx->error_msg, sizeof(ft_ctx->error_msg), "OOM tree"); longjmp(ft_ctx->error_jmp, FASTTREE_ERR_NOMEM); }

    tree->n_nodes  = n_nodes;
    tree->n_leaves = n_leaves_total;
    tree->root     = root;

    tree->nodes = (fasttree_node_t *)calloc(n_nodes, sizeof(fasttree_node_t));
    if (!tree->nodes) { free(tree); longjmp(ft_ctx->error_jmp, FASTTREE_ERR_NOMEM); }

    /* Count total children (NJ internal + extra children for duplicates) */
    int total_children = 0;
    for (i = n_unique; i < NJ->maxnode; i++)
      total_children += NJ->child[i].nChild;
    /* Unique leaves with duplicates become internal-like: original + dup children */
    for (i = 0; i < n_unique; i++) {
      int nDup = 0;
      int alnIdx = unique->alnNext[unique->uniqueFirst[i]];
      while (alnIdx >= 0) { nDup++; alnIdx = unique->alnNext[alnIdx]; }
      if (nDup > 0)
        total_children += nDup + 1; /* original leaf + duplicates */
    }
    tree->_children_buf = (int *)malloc(sizeof(int) * (total_children > 0 ? total_children : 1));
    if (!tree->_children_buf) { free(tree->nodes); free(tree); longjmp(ft_ctx->error_jmp, FASTTREE_ERR_NOMEM); }

    /* Calculate name buffer size for ALL leaves (unique + duplicates) */
    size_t name_buf_size = 0;
    for (i = 0; i < n_unique; i++) {
      int alnIdx = unique->uniqueFirst[i];
      while (alnIdx >= 0) {
        name_buf_size += strlen(aln->names[alnIdx]) + 1;
        alnIdx = unique->alnNext[alnIdx];
      }
    }
    tree->_name_buf = (char *)malloc(name_buf_size > 0 ? name_buf_size : 1);
    if (!tree->_name_buf) { free(tree->_children_buf); free(tree->nodes); free(tree); longjmp(ft_ctx->error_jmp, FASTTREE_ERR_NOMEM); }

    /* Fill nodes */
    int *child_ptr = tree->_children_buf;
    char *name_ptr = tree->_name_buf;
    int next_dup_id = NJ->maxnode;  /* IDs for duplicate leaf nodes */

    for (i = 0; i < NJ->maxnode; i++) {
      fasttree_node_t *node = &tree->nodes[i];
      node->id = i;
      node->parent = NJ->parent[i];
      node->branch_length = NJ->branchlength[i];
      node->support = NJ->support[i];

      if (i < n_unique) {
        /* Unique leaf — check for duplicates */
        int nDup = 0;
        int alnIdx = unique->alnNext[unique->uniqueFirst[i]];
        while (alnIdx >= 0) { nDup++; alnIdx = unique->alnNext[alnIdx]; }

        if (nDup == 0) {
          /* Simple leaf, no duplicates */
          node->is_leaf = 1;
          node->n_children = 0;
          node->children = NULL;
          int origIdx = unique->uniqueFirst[i];
          const char *origName = aln->names[origIdx];
          size_t nlen = strlen(origName);
          memcpy(name_ptr, origName, nlen + 1);
          node->name = name_ptr;
          name_ptr += nlen + 1;
        } else {
          /* Leaf with duplicates — becomes internal-like with children */
          node->is_leaf = 0;
          node->n_children = nDup + 1;
          node->children = child_ptr;
          node->name = NULL;

          /* First child: the original leaf (as a new dup node) */
          int origIdx = unique->uniqueFirst[i];
          int dupId = next_dup_id++;
          *child_ptr++ = dupId;
          fasttree_node_t *dn = &tree->nodes[dupId];
          dn->id = dupId;
          dn->parent = i;
          dn->branch_length = 0.0;
          dn->support = -1;
          dn->is_leaf = 1;
          dn->n_children = 0;
          dn->children = NULL;
          const char *origName = aln->names[origIdx];
          size_t nlen = strlen(origName);
          memcpy(name_ptr, origName, nlen + 1);
          dn->name = name_ptr;
          name_ptr += nlen + 1;

          /* Remaining children: duplicate leaves */
          alnIdx = unique->alnNext[origIdx];
          while (alnIdx >= 0) {
            dupId = next_dup_id++;
            *child_ptr++ = dupId;
            dn = &tree->nodes[dupId];
            dn->id = dupId;
            dn->parent = i;
            dn->branch_length = 0.0;
            dn->support = -1;
            dn->is_leaf = 1;
            dn->n_children = 0;
            dn->children = NULL;
            const char *dupName = aln->names[alnIdx];
            nlen = strlen(dupName);
            memcpy(name_ptr, dupName, nlen + 1);
            dn->name = name_ptr;
            name_ptr += nlen + 1;
            alnIdx = unique->alnNext[alnIdx];
          }
        }
      } else {
        /* NJ internal node */
        node->is_leaf = 0;
        node->name = NULL;
        node->n_children = NJ->child[i].nChild;
        node->children = child_ptr;
        int j;
        for (j = 0; j < NJ->child[i].nChild; j++)
          *child_ptr++ = NJ->child[i].child[j];
      }
    }

    *tree_out = tree;
  }

  /* Fill stats (capture values before arena destroy) */
  if (stats_out) {
    stats_out->n_unique_seqs  = unique->nUnique;
    stats_out->log_likelihood = final_loglk;
    stats_out->gamma_log_lk   = -1.0;
    stats_out->n_nni          = FT_nNNI;
    stats_out->n_spr          = FT_nSPR;
    stats_out->n_ml_nni       = FT_nML_NNI;
  }

  /* Destroy arena — reclaims all computation memory in one call.
     The output tree uses system malloc so it survives this.
     Individual FreeNJ/FreeUniquify/etc. calls are unnecessary since
     the arena owns all that memory. */
  ft_arena_destroy(&ft_ctx->arena);

  return FASTTREE_OK;
}

/* ── Tree to Newick ───────────────────────────────────────────────── */

/* Recursive DFS helper for Newick serialization */
static int _newick_ensure(char **buf, size_t *len, size_t *cap, size_t need) {
  if (*len + need <= *cap) return 1;
  size_t newcap = (*cap) * 2 + need;
  char *p = (char *)realloc(*buf, newcap);
  if (!p) { free(*buf); *buf = NULL; return 0; }
  *buf = p;
  *cap = newcap;
  return 1;
}

static void _newick_recurse(const fasttree_tree_t *tree, int node_id,
                            int show_support, char **buf, size_t *len, size_t *cap) {
  if (!*buf) return;  /* propagate OOM */

  const fasttree_node_t *node = &tree->nodes[node_id];

  if (!_newick_ensure(buf, len, cap, 256)) return;

  if (node->is_leaf) {
    *len += snprintf(*buf + *len, *cap - *len, "%s", node->name ? node->name : "");
  } else {
    *len += snprintf(*buf + *len, *cap - *len, "(");
    int j;
    for (j = 0; j < node->n_children; j++) {
      if (!*buf) return;  /* propagate OOM from child */
      if (j > 0) {
        if (!_newick_ensure(buf, len, cap, 1)) return;
        (*buf)[(*len)++] = ',';
      }
      _newick_recurse(tree, node->children[j], show_support, buf, len, cap);
    }
    if (!*buf) return;
    if (!_newick_ensure(buf, len, cap, 256)) return;
    *len += snprintf(*buf + *len, *cap - *len, ")");
    if (show_support && node->parent >= 0 && node->support >= 0) {
      *len += snprintf(*buf + *len, *cap - *len, "%.3f", node->support);
    }
  }

  if (node->parent >= 0) {
    if (!_newick_ensure(buf, len, cap, 64)) return;
#ifdef USE_DOUBLE
    *len += snprintf(*buf + *len, *cap - *len, ":%.9f", node->branch_length);
#else
    *len += snprintf(*buf + *len, *cap - *len, ":%.5f", node->branch_length);
#endif
  }
}

char *fasttree_tree_to_newick(const fasttree_tree_t *tree, int show_support) {
  if (tree == NULL || tree->nodes == NULL) return NULL;

  size_t cap = 4096;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) return NULL;

  _newick_recurse(tree, tree->root, show_support, &buf, &len, &cap);
  if (!buf) return NULL;

  /* Append ;\n */
  if (len + 3 > cap) {
    cap = len + 3;
    buf = (char *)realloc(buf, cap);
    if (!buf) return NULL;
  }
  buf[len++] = ';';
  buf[len++] = '\n';
  buf[len] = '\0';

  return buf;
}

/* ── Tree free ────────────────────────────────────────────────────── */

void fasttree_tree_free(fasttree_tree_t *tree) {
  if (tree == NULL) return;
  free(tree->_children_buf);
  free(tree->_name_buf);
  free(tree->nodes);
  free(tree);
}

/* ── Error handling ───────────────────────────────────────────────── */

const char *fasttree_strerror(int error_code) {
  switch (error_code) {
    case FASTTREE_OK:                return "Success";
    case FASTTREE_ERR_NOMEM:         return "Out of memory";
    case FASTTREE_ERR_INVALID_INPUT: return "Invalid input";
    case FASTTREE_ERR_PARSE:         return "Parse error";
    case FASTTREE_ERR_INTERNAL:      return "Internal error";
    case FASTTREE_ERR_INVALID_CONFIG:return "Invalid configuration";
    case FASTTREE_ERR_CANCELLED:     return "Cancelled";
    default:                         return "Unknown error";
  }
}

const char *fasttree_last_error(const fasttree_ctx_t *ctx) {
  if (ctx == NULL) return "NULL context";
  return ctx->error_msg;
}
