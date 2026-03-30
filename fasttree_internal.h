/*
 * fasttree_internal.h — Internal header for the FastTree library.
 *
 * Contains type definitions (moved from FastTree.c), the context struct
 * definition, macro aliases for global-to-context migration, and the
 * arena allocator.
 *
 * This header is NOT part of the public API.  Only fasttree_core.c,
 * fasttree_api.c should include it.
 */

#ifndef FASTTREE_INTERNAL_H
#define FASTTREE_INTERNAL_H

#include "fasttree.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <ctype.h>
#include <setjmp.h>
#include <sys/time.h>

#ifdef TRACK_MEMORY
#include <malloc.h>
#endif

#ifdef OPENMP
#include <omp.h>
#endif

/* ── Precision ───────────────────────────────────────────────────── */

#ifndef USE_SINGLE
#ifndef USE_DOUBLE
#define USE_DOUBLE
#endif
#endif

#ifdef USE_DOUBLE
typedef double numeric_t;
#define ScanNumericSpec "%lf"
#else
typedef float numeric_t;
#define ScanNumericSpec "%f"
#endif

/* SSE3 only used in single-precision mode */
#if defined(__SSE__) && defined(USE_SINGLE) && !defined(NO_SSE)
#define USE_SSE3
#endif

#ifdef USE_SSE3
#define ALIGNED __attribute__((aligned(16)))
#define IS_ALIGNED(X) ((((unsigned long)(X)) & 15L) == 0L)
#include <xmmintrin.h>
#else
#define ALIGNED
#define IS_ALIGNED(X) 1
#endif

/* ── Constants ───────────────────────────────────────────────────── */

#define MAXCODES 20
#define NOCODE 127
#define BUFFER_SIZE 5000

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif
#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

/* ── RNG constants ───────────────────────────────────────────────── */

#define KK 100
#define LL 37
#define MM (1L<<30)
#define mod_diff(x,y) (((x)-(y))&(MM-1))
#define TT 70
#define is_odd(x) ((x)&1)
#define QUALITY 1009

/* ── Type definitions (moved from FastTree.c) ────────────────────── */

typedef struct {
  int nPos;
  int nSeq;
  char **names;
  char **seqs;
  int nSaved; /* actual allocated size of names and seqs */
} alignment_t;

typedef struct {
  /* alignment profile */
  numeric_t *weights;
  unsigned char *codes;
  numeric_t *vectors;       /* NULL if no non-constant positions */
  int nVectors;
  numeric_t *codeDist;      /* Optional -- distance to each code at each position */

  /* constraint profile */
  int *nOn;
  int *nOff;
} profile_t;

typedef struct {
  int i, j;
  numeric_t weight;
  numeric_t dist;
  numeric_t criterion;
} besthit_t;

typedef struct {
  int nChild;
  int child[3];
} children_t;

typedef struct {
  numeric_t distances[MAXCODES][MAXCODES];
  numeric_t eigeninv[MAXCODES][MAXCODES];
  numeric_t eigenval[MAXCODES];
  numeric_t eigentot[MAXCODES];
  numeric_t codeFreq[MAXCODES][MAXCODES];
  numeric_t gapFreq[MAXCODES];
} distance_matrix_t;

typedef struct {
  numeric_t stat[MAXCODES];
  numeric_t statinv[MAXCODES];
  numeric_t codeFreq[NOCODE+1][MAXCODES];
  numeric_t eigeninv[MAXCODES][MAXCODES];
  numeric_t eigeninvT[MAXCODES][MAXCODES];
  numeric_t eigenval[MAXCODES];
  numeric_t nearP[MAXCODES][MAXCODES];
  numeric_t nearFreq[MAXCODES][MAXCODES];
} transition_matrix_t;

typedef struct {
  int nRateCategories;
  numeric_t *rates;
  unsigned int *ratecat;
} rates_t;

typedef struct {
  int nSeq;
  int nPos;
  char **seqs;
  distance_matrix_t *distance_matrix;
  transition_matrix_t *transmat;
  int nConstraints;
  char **constraintSeqs;

  int maxnode;
  int maxnodes;
  profile_t **profiles;
  numeric_t *diameter;
  numeric_t *varDiameter;
  numeric_t *selfdist;
  numeric_t *selfweight;

  profile_t *outprofile;
  double totdiam;

  numeric_t *outDistances;
  int *nOutDistActive;

  int root;
  int *parent;
  children_t *child;
  numeric_t *branchlength;
  numeric_t *support;

  rates_t rates;
} NJ_t;

typedef struct {
  int nSeq;
  int nUnique;
  int *uniqueFirst;
  int *alnNext;
  int *alnToUniq;
  char **uniqueSeq;
} uniquify_t;

typedef enum {ABvsCD, ACvsBD, ADvsBC} nni_t;

typedef struct {
  int nodes[2];
  double deltaLength;
} spr_step_t;

typedef struct {
  int j;
  numeric_t dist;
} hit_t;

typedef struct {
  int nHits;
  hit_t *hits;
  int hitSource;
  int age;
} top_hits_list_t;

typedef struct {
  int m;
  int q;
  int maxnodes;
  top_hits_list_t *top_hits_lists;
  hit_t *visible;
  int nTopVisible;
  int *topvisible;
  int topvisibleAge;
#ifdef OPENMP
  omp_lock_t *locks;
#endif
} top_hits_t;

typedef struct {
  int age;
  int subtreeAge;
  double delta;
  double support;
} nni_stats_t;

typedef struct {
  int nBadSplits;
  int nConstraintViolations;
  int nBadBoth;
  int nSplits;
  double dWorstDeltaUnconstrained;
  double dWorstDeltaConstrained;
} SplitCount_t;

typedef bool *traversal_t;

typedef struct {
  char *string;
  int nCount;
  int first;
} hashbucket_t;

typedef struct {
  int nBuckets;
  hashbucket_t *buckets;
} hashstrings_t;

typedef int hashiterator_t;

typedef enum {qAB, qAC, qAD, qBC, qBD, qCD} quartet_pair_t;

typedef enum {LEN_A, LEN_B, LEN_C, LEN_D, LEN_I} quartet_length_t;

typedef struct {
  int nPos;
  transition_matrix_t *transmat;
  rates_t *rates;
  int nEval;
  profile_t *pair1;
  profile_t *pair2;
  struct fasttree_ctx *ft_ctx;  /* for callbacks passed to brent/onedimenmin */
} quartet_opt_t;

typedef struct {
  NJ_t *NJ;
  double freq[4];
  double rates[6];
  int iRate;
  FILE *fpLog;
  struct fasttree_ctx *ft_ctx;  /* for callbacks passed to brent/onedimenmin */
} gtr_opt_t;

typedef struct {
  double mult;
  double alpha;
  int nPos;
  int nRateCats;
  numeric_t *rates;
  double *site_loglk;
  struct fasttree_ctx *ft_ctx;  /* for callbacks passed to brent/onedimenmin */
} siteratelk_t;

/* ── Arena allocator ─────────────────────────────────────────────── */

/*
 * Simple arena: linked list of large blocks.
 * All allocations during fasttree_build() go through the arena.
 * On longjmp error, fasttree_destroy() frees the entire arena.
 */

#define FT_ARENA_DEFAULT_BLOCK_SIZE (1 << 20)  /* 1 MiB */

typedef struct ft_arena_block {
  struct ft_arena_block *next;
  size_t size;
  size_t used;
  /* data follows (flexible array member) */
  char data[];
} ft_arena_block_t;

typedef struct {
  ft_arena_block_t *head;
  size_t block_size;
  /* Oversized allocations (> block_size) get their own block at the
     head of the same linked list.  No separate tracking needed since
     ft_arena_destroy walks the entire list. */
} ft_arena_t;

void  ft_arena_init(ft_arena_t *arena);
void *ft_arena_alloc(ft_arena_t *arena, size_t size);
void *ft_arena_realloc(ft_arena_t *arena, void *ptr, size_t old_size, size_t new_size);
void  ft_arena_destroy(ft_arena_t *arena);

/* ── Context struct ──────────────────────────────────────────────── */

struct fasttree_ctx {
  /* --- Error handling --- */
  jmp_buf error_jmp;
  int     error_code;
  char    error_msg[512];
  int     cancelled;

  /* --- Arena allocator --- */
  ft_arena_t arena;

  /* --- Callbacks (copied from config) --- */
  int   (*progress_callback)(const char *stage, double frac_done, void *user_data);
  void   *progress_user_data;
  void  (*log_callback)(const char *msg, void *user_data);
  void   *log_user_data;

  /* --- Algorithm options --- */
  int    verbose;
  int    showProgress;
  int    slow;
  int    fastest;
  bool   useTopHits2nd;
  int    bionj;
  double tophitsMult;
  double tophitsClose;
  double topvisibleMult;
  double tophitsRefresh;
  double tophits2Mult;
  int    tophits2Safety;
  double tophits2Refresh;
  double staleOutLimit;
  double fResetOutProfile;
  int    nResetOutProfile;
  int    nCodes;
  bool   useMatrix;
  bool   logdist;
  double pseudoWeight;
  double constraintWeight;
  double MEMinDelta;
  bool   fastNNI;
  bool   bGammaLogLk;   /* b-prefix because RescaleGammaLogLk() has a local `gammaLogLk` */

  /* ML options */
  int    mlAccuracy;
  double closeLogLkLimit;
  double treeLogLkDelta;
  bool   exactML;
  double approxMLminf;
  double approxMLminratio;
  double approxMLnearT;

  /* Character sets (mutable — assigned based on seq type) */
  unsigned char *codesString;
  unsigned char *codesStringAA;
  unsigned char *codesStringNT;

  /* Character-to-code mapping — placeholder for Phase 3c.
     Wired up from former static locals in SeqToProfile. */
  unsigned char charToCode[256];
  int           charToCodeSet;

  /* --- Performance counters --- */
  long profileOps;
  long outprofileOps;
  long seqOps;
  long profileAvgOps;
  long nHillBetter;
  long nCloseUsed;
  long nClose2Used;
  long nRefreshTopHits;
  long nVisibleUpdate;
  long nNNI;
  long nSPR;
  long nML_NNI;
  long nSuboptimalSplits;
  /* nSuboptimalConstrained and nConstraintViolations (global counters) were
     declared but never used in the original FastTree.c.  Omitted here. */
  long nProfileFreqAlloc;
  long nProfileFreqAvoid;
  long szAllAlloc;
  long mymallocUsed;
  long maxmallocHeap;
  long nLkCompute;
  long nPosteriorCompute;
  long nAAPosteriorExact;
  long nAAPosteriorRough;
  long nStarTests;

  /* --- RNG state (Knuth TAOCP) ---
     INVARIANT: ran_arr_ptr must point into this same struct instance
     (either &ran_arr_dummy or into ran_arr_buf).  This struct must NOT
     be memcpy'd/moved after initialization — ran_arr_ptr would dangle.
     fasttree_create() must set ran_arr_ptr = &ctx->ran_arr_dummy. */
  long ran_x[KK];
  long ran_arr_buf[QUALITY];
  long ran_arr_dummy;
  long ran_arr_started;
  long *ran_arr_ptr;

  /* --- Progress/timing state — placeholder for Phase 3c.
     Wired up from former static locals in ProgressReport(). --- */
  bool         time_set;
  struct timeval time_last;
  struct timeval time_begin;

  /* --- Buffers (moved from static locals for thread safety) --- */
  char tree_token_buf[BUFFER_SIZE]; /* ReadTreeToken return buffer */

  /* --- OpenMP --- */
  int n_threads;

  /* --- Output options --- */
  int bQuote;

  /* --- Config snapshot (for fasttree_build) --- */
  int     nni_rounds;
  int     spr_rounds;
  int     ml_nni_rounds;
  int     n_rate_cats;
  int     n_bootstrap;
  int64_t seed;
  int     model;             /* fasttree_model_t enum value */
  int     gtr_from_alignment;
  double  gtr_rates[6];
  double  gtr_freq[4];

  /* --- Starting tree (optional, from config) --- */
  const char *start_newick;
};

/* ── Macro aliases for global-to-context migration ───────────────── */
/*
 * These macros allow the existing function bodies to remain textually
 * unchanged.  E.g. code that reads `verbose` now resolves to
 * `ft_ctx->verbose`.  Function signatures must add
 * `fasttree_ctx_t *ft_ctx` as the first parameter.
 */

#define FT_verbose          (ft_ctx->verbose)
#define FT_showProgress     (ft_ctx->showProgress)
#define FT_slow             (ft_ctx->slow)
#define FT_fastest          (ft_ctx->fastest)
#define FT_useTopHits2nd    (ft_ctx->useTopHits2nd)
#define FT_bionj            (ft_ctx->bionj)
#define FT_tophitsMult      (ft_ctx->tophitsMult)
#define FT_tophitsClose     (ft_ctx->tophitsClose)
#define FT_topvisibleMult   (ft_ctx->topvisibleMult)
#define FT_tophitsRefresh   (ft_ctx->tophitsRefresh)
#define FT_tophits2Mult     (ft_ctx->tophits2Mult)
#define FT_tophits2Safety   (ft_ctx->tophits2Safety)
#define FT_tophits2Refresh  (ft_ctx->tophits2Refresh)
#define FT_staleOutLimit    (ft_ctx->staleOutLimit)
#define FT_fResetOutProfile (ft_ctx->fResetOutProfile)
#define FT_nResetOutProfile (ft_ctx->nResetOutProfile)
#define FT_nCodes           (ft_ctx->nCodes)
#define FT_useMatrix        (ft_ctx->useMatrix)
#define FT_logdist          (ft_ctx->logdist)
#define FT_pseudoWeight     (ft_ctx->pseudoWeight)
#define FT_constraintWeight (ft_ctx->constraintWeight)
#define FT_MEMinDelta       (ft_ctx->MEMinDelta)
#define FT_fastNNI          (ft_ctx->fastNNI)
#define FT_gammaLogLk       (ft_ctx->bGammaLogLk)
#define FT_mlAccuracy       (ft_ctx->mlAccuracy)
#define FT_closeLogLkLimit  (ft_ctx->closeLogLkLimit)
#define FT_treeLogLkDelta   (ft_ctx->treeLogLkDelta)
#define FT_exactML          (ft_ctx->exactML)
#define FT_approxMLminf     (ft_ctx->approxMLminf)
#define FT_approxMLminratio (ft_ctx->approxMLminratio)
#define FT_approxMLnearT    (ft_ctx->approxMLnearT)
#define FT_codesString      (ft_ctx->codesString)
#define FT_codesStringAA    (ft_ctx->codesStringAA)
#define FT_codesStringNT    (ft_ctx->codesStringNT)
#define FT_profileOps       (ft_ctx->profileOps)
#define FT_outprofileOps    (ft_ctx->outprofileOps)
#define FT_seqOps           (ft_ctx->seqOps)
#define FT_profileAvgOps    (ft_ctx->profileAvgOps)
#define FT_nHillBetter      (ft_ctx->nHillBetter)
#define FT_nCloseUsed       (ft_ctx->nCloseUsed)
#define FT_nClose2Used      (ft_ctx->nClose2Used)
#define FT_nRefreshTopHits  (ft_ctx->nRefreshTopHits)
#define FT_nVisibleUpdate   (ft_ctx->nVisibleUpdate)
#define FT_nNNI             (ft_ctx->nNNI)
#define FT_nSPR             (ft_ctx->nSPR)
#define FT_nML_NNI          (ft_ctx->nML_NNI)
#define FT_nSuboptimalSplits (ft_ctx->nSuboptimalSplits)
/* FT_nSuboptimalConstrained and FT_nConstraintViolations (global counters)
   omitted — were declared but never used in original FastTree.c. */
#define FT_nProfileFreqAlloc (ft_ctx->nProfileFreqAlloc)
#define FT_nProfileFreqAvoid (ft_ctx->nProfileFreqAvoid)
#define FT_szAllAlloc       (ft_ctx->szAllAlloc)
#define FT_mymallocUsed     (ft_ctx->mymallocUsed)
#define FT_maxmallocHeap    (ft_ctx->maxmallocHeap)
#define FT_nLkCompute       (ft_ctx->nLkCompute)
#define FT_nPosteriorCompute (ft_ctx->nPosteriorCompute)
#define FT_nAAPosteriorExact (ft_ctx->nAAPosteriorExact)
#define FT_nAAPosteriorRough (ft_ctx->nAAPosteriorRough)
#define FT_nStarTests       (ft_ctx->nStarTests)

/* RNG macros */
#define FT_ran_x            (ft_ctx->ran_x)
#define FT_ran_arr_buf      (ft_ctx->ran_arr_buf)
#define FT_ran_arr_dummy    (ft_ctx->ran_arr_dummy)
#define FT_ran_arr_started  (ft_ctx->ran_arr_started)
#define FT_ran_arr_ptr      (ft_ctx->ran_arr_ptr)

/* Progress timing (moved from static locals in ProgressReport) */
#define FT_time_set         (ft_ctx->time_set)
#define FT_time_last        (ft_ctx->time_last)
#define FT_time_begin       (ft_ctx->time_begin)

/* Character encoding (moved from static locals in SeqToProfile) */
#define FT_charToCode       (ft_ctx->charToCode)
#define FT_charToCodeSet    (ft_ctx->charToCodeSet)


/* ── Read-only data externs (stay as file-scope in fasttree_core.c) ── */

extern distance_matrix_t matrixBLOSUM45;
extern double matrixJTT92[MAXCODES][MAXCODES];
extern double statJTT92[MAXCODES];
extern double matrixLG08[MAXCODES][MAXCODES];
extern double statLG08[MAXCODES];
extern double matrixWAG01[MAXCODES][MAXCODES];
extern double statWAG01[MAXCODES];

/* ── Internal function declarations ──────────────────────────────── */

/* Context initialization (called by main and fasttree_create) */
void fasttree_ctx_init(fasttree_ctx_t *c);

/* Logging helpers */
void ft_log(fasttree_ctx_t *ft_ctx, const char *fmt, ...);
int  ft_progress(fasttree_ctx_t *ft_ctx, const char *stage, double frac_done);

/* ── Core function declarations (fasttree_core.c) ─────────────────── */
/* These are used by fasttree_api.c to implement fasttree_build(). */

/* Memory */
void *mymalloc(fasttree_ctx_t *ft_ctx, size_t sz);
void *myfree(fasttree_ctx_t *ft_ctx, void *, size_t sz);
void *mymemdup(fasttree_ctx_t *ft_ctx, void *data, size_t sz);
void *myrealloc(fasttree_ctx_t *ft_ctx, void *data, size_t szOld, size_t szNew, bool bCopy);

/* Progress */
void ProgressReport(fasttree_ctx_t *ft_ctx, char *format, int i1, int i2, int i3, int i4);

/* Alignment */
alignment_t *AlignmentFromMemory(fasttree_ctx_t *ft_ctx,
                                 const char **names, const char **seqs,
                                 int nSeq, int nPos);
alignment_t *ReadAlignment(fasttree_ctx_t *ft_ctx, FILE *fp, bool bQuote);
alignment_t *FreeAlignment(fasttree_ctx_t *ft_ctx, alignment_t *);
void FreeAlignmentSeqs(fasttree_ctx_t *ft_ctx, alignment_t *);

/* Hashing */
hashstrings_t *MakeHashtable(fasttree_ctx_t *ft_ctx, char **strings, int nStrings);
hashstrings_t *FreeHashtable(fasttree_ctx_t *ft_ctx, hashstrings_t *hash);
hashiterator_t FindMatch(fasttree_ctx_t *ft_ctx, hashstrings_t *hash, char *string);
int HashCount(fasttree_ctx_t *ft_ctx, hashstrings_t *hash, hashiterator_t hi);

/* Uniquify */
uniquify_t *UniquifyAln(fasttree_ctx_t *ft_ctx, alignment_t *aln);
uniquify_t *FreeUniquify(fasttree_ctx_t *ft_ctx, uniquify_t *);

/* Distance/transition matrices */
void SetupDistanceMatrix(fasttree_ctx_t *ft_ctx, distance_matrix_t *);
transition_matrix_t *CreateTransitionMatrix(fasttree_ctx_t *ft_ctx,
                                            double matrix[MAXCODES][MAXCODES],
                                            double stat[MAXCODES]);
transition_matrix_t *CreateGTR(fasttree_ctx_t *ft_ctx, double *gtrrates, double *gtrfreq);
distance_matrix_t *TransMatToDistanceMat(fasttree_ctx_t *ft_ctx, transition_matrix_t *transmat);

/* NJ core */
NJ_t *InitNJ(fasttree_ctx_t *ft_ctx, char **sequences, int nSeqs, int nPos,
             char **constraintSeqs, int nConstraints,
             distance_matrix_t *, transition_matrix_t *);
NJ_t *FreeNJ(fasttree_ctx_t *ft_ctx, NJ_t *NJ);
void FastNJ(fasttree_ctx_t *ft_ctx, NJ_t *NJ);
void ReliabilityNJ(fasttree_ctx_t *ft_ctx, NJ_t *NJ, int nBootstrap);

/* NNI/SPR */
int NNI(fasttree_ctx_t *ft_ctx, NJ_t *NJ, int iRound, int nRounds, bool useML,
        nni_stats_t *stats, double *maxDeltaCriterion);
nni_stats_t *InitNNIStats(fasttree_ctx_t *ft_ctx, NJ_t *NJ);
nni_stats_t *FreeNNIStats(fasttree_ctx_t *ft_ctx, nni_stats_t *, NJ_t *NJ);
void SPR(fasttree_ctx_t *ft_ctx, NJ_t *NJ, int maxSPRLength, int iRound, int nRounds);
void UpdateBranchLengths(fasttree_ctx_t *ft_ctx, NJ_t *NJ);

/* ML */
void OptimizeAllBranchLengths(fasttree_ctx_t *ft_ctx, NJ_t *NJ);
double TreeLogLk(fasttree_ctx_t *ft_ctx, NJ_t *NJ, double *site_loglk);
void SetMLRates(fasttree_ctx_t *ft_ctx, NJ_t *NJ, int nRateCategories);
numeric_t *MLSiteRates(fasttree_ctx_t *ft_ctx, int nRateCategories);
double *MLSiteLikelihoodsByRate(fasttree_ctx_t *ft_ctx, NJ_t *NJ, numeric_t *rates, int nRateCategories);
double RescaleGammaLogLk(fasttree_ctx_t *ft_ctx, int nPos, int nRateCats,
                         numeric_t *rates, double *site_loglk, FILE *fpLog);
void SetMLGtr(fasttree_ctx_t *ft_ctx, NJ_t *NJ, double *gtrfreq, FILE *fpLog);
void RecomputeProfiles(fasttree_ctx_t *ft_ctx, NJ_t *NJ, distance_matrix_t *dmat);

/* Support */
void TestSplitsMinEvo(fasttree_ctx_t *ft_ctx, NJ_t *NJ, SplitCount_t *splitcount);
void TestSplitsML(fasttree_ctx_t *ft_ctx, NJ_t *NJ, SplitCount_t *splitcount, int nBootstrap);

/* RNG */
void ran_start(fasttree_ctx_t *ft_ctx, long seed);

/* Constraints */
char **AlnToConstraints(fasttree_ctx_t *ft_ctx, alignment_t *constraints,
                        uniquify_t *unique, hashstrings_t *hashnames);

/* Tree I/O */
void ReadTree(fasttree_ctx_t *ft_ctx, NJ_t *NJ, uniquify_t *unique,
              hashstrings_t *hashnames, FILE *fpInTree);
void PrintNJ(fasttree_ctx_t *ft_ctx, FILE *, NJ_t *NJ, char **names,
             uniquify_t *unique, bool bShowSupport, bool bQuoteNames);

#endif /* FASTTREE_INTERNAL_H */
