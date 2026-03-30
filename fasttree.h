/*
 * fasttree.h — Public API for the FastTree phylogenetic tree library.
 *
 * Copyright (C) 2008-2015 The Regents of the University of California
 * (original FastTree code by Morgan N. Price)
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Thread safety: Each fasttree_ctx_t is independent.  It is safe to use
 * different contexts concurrently from different threads.  A single
 * context must not be used from multiple threads simultaneously.
 *
 * Context reuse: A context may be reused for multiple fasttree_build()
 * calls.  After an error return, the context is reset and may be called
 * again with different input.  The error message from fasttree_last_error()
 * remains valid until the next API call on the same context, or until
 * fasttree_destroy().
 */

#ifndef FASTTREE_H
#define FASTTREE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Symbol visibility ───────────────────────────────────────────── */

#ifndef FASTTREE_API
#  if defined(_WIN32) && defined(FASTTREE_BUILDING_DLL)
#    define FASTTREE_API __declspec(dllexport)
#  elif defined(_WIN32)
#    define FASTTREE_API __declspec(dllimport)
#  elif defined(__GNUC__) && __GNUC__ >= 4
#    define FASTTREE_API __attribute__((visibility("default")))
#  else
#    define FASTTREE_API
#  endif
#endif

/* ── Version ─────────────────────────────────────────────────────── */

#define FASTTREE_VERSION_MAJOR  3
#define FASTTREE_VERSION_MINOR  0
#define FASTTREE_VERSION_PATCH  0
#define FASTTREE_VERSION_STRING "3.0.0"

/* ── Error codes ─────────────────────────────────────────────────── */

#define FASTTREE_OK                   0
#define FASTTREE_ERR_NOMEM           -1
#define FASTTREE_ERR_INVALID_INPUT   -2
#define FASTTREE_ERR_PARSE           -3
#define FASTTREE_ERR_INTERNAL        -4
#define FASTTREE_ERR_INVALID_CONFIG  -5
#define FASTTREE_ERR_CANCELLED       -6

/* ── Opaque context ──────────────────────────────────────────────── */

typedef struct fasttree_ctx fasttree_ctx_t;

/* ── Enums ───────────────────────────────────────────────────────── */

typedef enum {
    FASTTREE_SEQ_AUTO,
    FASTTREE_SEQ_PROTEIN,
    FASTTREE_SEQ_NUCLEOTIDE
} fasttree_seqtype_t;

typedef enum {
    FASTTREE_MODEL_AUTO,   /* JTT for protein, JC for nucleotide */
    FASTTREE_MODEL_JTT,
    FASTTREE_MODEL_LG,
    FASTTREE_MODEL_WAG,
    FASTTREE_MODEL_JC,
    FASTTREE_MODEL_GTR
} fasttree_model_t;

/* ── Configuration ───────────────────────────────────────────────── */

/*
 * All int fields that represent booleans use 0=false, nonzero=true.
 * This avoids ABI issues with the C _Bool / C++ bool type whose size
 * is implementation-defined.
 *
 * Always initialize with fasttree_config_init() which sets struct_size
 * and all defaults.  New fields added in future versions will be
 * appended at the end; fasttree_create() checks struct_size to detect
 * version mismatches.
 */

typedef struct {
    /* ABI versioning — must be first field. Set by fasttree_config_init(). */
    size_t struct_size;

    /* Sequence and model */
    fasttree_seqtype_t seq_type;       /* AUTO = infer from data */
    fasttree_model_t   model;          /* AUTO = JTT for protein, JC for nucleotide */

    /* GTR parameters (only when model == FASTTREE_MODEL_GTR).
       If gtr_from_alignment is nonzero (the default for GTR), rates and
       frequencies are estimated from the alignment and gtr_rates/gtr_freq
       are ignored.  Set gtr_from_alignment=0 and fill in gtr_rates/gtr_freq
       to supply explicit parameters. */
    double gtr_rates[6];               /* ac, ag, at, cg, ct, gt */
    double gtr_freq[4];                /* A, C, G, T frequencies */
    int    gtr_from_alignment;         /* nonzero = estimate from data (default) */

    /* Tree search */
    int  nni_rounds;                   /* -1 = auto */
    int  spr_rounds;                   /* default 2 */
    int  ml_nni_rounds;                /* -1 = auto, 0 = disable ML NNI */
    int  n_rate_cats;                  /* default 20; number of CAT rate categories */
    int  slow;                         /* nonzero = exhaustive NJ search (much slower) */
    int  fastest;                      /* nonzero = fastest heuristics, less accurate */

    /* Support values */
    int  n_bootstrap;                  /* default 1000; SH-like support resamples */
    int  gamma_log_lk;                 /* nonzero = report gamma likelihood */

    /* Reproducibility */
    int64_t seed;                      /* default 314159 */

    /* Heuristic parameters */
    int    use_top_hits;               /* nonzero = use top-hits heuristic (default 1) */
    double top_hits_mult;              /* 0 = compare all nodes; default 1.0 */
    int    use_bionj;                  /* nonzero = weighted (BIONJ) joins; default 0 */
    double pseudo_weight;              /* pseudocount weight; default 0.0 (off) */
    double constraint_weight;          /* penalty for constraint violations; default 100.0 */
    int    ml_accuracy;                /* ML branch-length optimization rounds; default 1 */

    /* Starting tree (optional).
       If non-NULL, parse this Newick string as the starting topology.
       Branch lengths in the starting tree are ignored.
       Equivalent to the CLI -intree option. */
    const char *start_newick;

    /* Output */
    int  quote_names;                  /* nonzero = quote names in Newick output */

    /* OpenMP thread control */
    int  n_threads;                    /* 0 = use OMP_NUM_THREADS env, >0 = explicit */

    /* Progress callback.
       Called periodically during computation.  `stage` is an informational
       string (e.g. "NJ", "ME-NNI", "ML-NNI") — do not branch on its value
       as the set of stages may change between versions.  `frac_done` is in
       [0,1] within the current stage but is not globally monotonic across
       stages.
       Return 0 to continue, nonzero to cancel
       (fasttree_build returns FASTTREE_ERR_CANCELLED). */
    int  (*progress_callback)(const char *stage, double frac_done, void *user_data);
    void  *progress_user_data;

    /* Log callback.  Receives verbose/debug messages (may be multi-line). */
    void (*log_callback)(const char *msg, void *user_data);
    void  *log_user_data;

    /* Custom allocator (optional).
       If alloc_fn is non-NULL, the library uses it instead of malloc/free
       for all internal computation memory.  The tree output struct returned
       by fasttree_build is always allocated with system malloc.
       alloc_fn must return 16-byte-aligned memory or NULL on failure.
       free_fn must accept pointers returned by alloc_fn. */
    void *(*alloc_fn)(size_t size, void *user_data);
    void  (*free_fn)(void *ptr, void *user_data);
    void  *alloc_user_data;
} fasttree_config_t;

/* ── Build statistics ────────────────────────────────────────────── */

/* Returned by fasttree_build alongside the tree.  All fields are
   informational; the struct is owned by the caller and has no
   pointers that need freeing. */
typedef struct {
    int    n_unique_seqs;    /* sequences after deduplication */
    double log_likelihood;   /* final tree log-likelihood (-1 if ML disabled) */
    double gamma_log_lk;     /* gamma log-likelihood (-1 if not computed) */
    int    n_nni;            /* total ME-NNI topology changes */
    int    n_spr;            /* total SPR topology changes */
    int    n_ml_nni;         /* total ML-NNI topology changes */
} fasttree_stats_t;

/* ── Tree output ─────────────────────────────────────────────────── */

/*
 * All public-facing numeric values are double regardless of whether
 * the library was compiled with USE_DOUBLE or USE_SINGLE.
 *
 * children is a pointer into the tree's internal _children_buf.
 * Do not free or reallocate it directly; use fasttree_tree_free()
 * to release the entire tree.
 *
 * Duplicate sequences appear as zero-branch-length leaf children
 * under their unique representative, so a node may have more than
 * 3 children.
 */

typedef struct {
    int          id;
    int          parent;        /* -1 for root */
    int         *children;      /* do not free directly */
    int          n_children;    /* 0=leaf, 2-3=internal, may be >3 with duplicates */
    double       branch_length;
    double       support;       /* -1 if not computed */
    const char  *name;          /* leaf name, or NULL for internal nodes */
    int          is_leaf;       /* nonzero = leaf node */
} fasttree_node_t;

typedef struct {
    int               n_nodes;
    int               n_leaves;
    int               root;          /* index into nodes[] */
    fasttree_node_t  *nodes;
    /* Internal backing stores — do not access directly. */
    int              *_children_buf;
    char             *_name_buf;
} fasttree_tree_t;

/* ── API functions ───────────────────────────────────────────────── */

/* Fill config with defaults matching standard FastTree behavior.
   Must be called before modifying any config fields. */
FASTTREE_API void            fasttree_config_init(fasttree_config_t *config);

/* Create a new context from the given config.  Returns NULL on OOM. */
FASTTREE_API fasttree_ctx_t *fasttree_create(const fasttree_config_t *config);

/* Free a context and all memory it owns. */
FASTTREE_API void            fasttree_destroy(fasttree_ctx_t *ctx);

/* Build a tree from an in-memory alignment.
   names[i] and seqs[i] for i in [0, nSeq).  Each seq must have nPos
   characters.  On success, *tree_out is set to a newly allocated tree
   (caller frees with fasttree_tree_free).  If stats_out is non-NULL,
   it is filled with build statistics.
   Returns FASTTREE_OK or an error code. */
FASTTREE_API int             fasttree_build(fasttree_ctx_t *ctx,
                                            const char **names, const char **seqs,
                                            int nSeq, int nPos,
                                            fasttree_tree_t **tree_out,
                                            fasttree_stats_t *stats_out);

/* Serialize a tree to Newick format.  Returns malloc'd string (caller
   frees with free()).  Branch-length precision matches the compile-time
   numeric precision (9 digits for double, 5 for float).  Support values
   use 3 decimal places. */
FASTTREE_API char           *fasttree_tree_to_newick(const fasttree_tree_t *tree,
                                                     int show_support);

/* Free a tree returned by fasttree_build. */
FASTTREE_API void            fasttree_tree_free(fasttree_tree_t *tree);

/* Human-readable string for an error code.  Returns "Unknown error" for
   unrecognized codes.  The returned string is a static constant. */
FASTTREE_API const char     *fasttree_strerror(int error_code);

/* Detailed error message from the last failed operation on this context.
   The returned pointer is valid until the next fasttree_build() call on
   this context, or until fasttree_destroy().  Copy the string if you
   need it to outlive the context. */
FASTTREE_API const char     *fasttree_last_error(const fasttree_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* FASTTREE_H */
