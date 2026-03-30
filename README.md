# FastTree

FastTree infers approximately-maximum-likelihood phylogenetic trees from alignments of nucleotide or protein sequences. It can handle alignments with up to a million sequences in a reasonable amount of time and memory.

FastTree is available as both a **command-line tool** and an **embeddable C library** (`libfasttree`).

## Building

```bash
make            # builds FastTree binary + libfasttree.a + libfasttree.so
make OMP=1      # with OpenMP multi-threading
make test       # run ground truth tests (bit-identical output verification)
make test_api   # run library API smoke tests
```

Requires a C99 compiler (gcc or clang) and `-lm`.

## Command-Line Usage

```bash
# Nucleotide alignment (FASTA or interleaved PHYLIP)
FastTree -nt alignment.fasta > tree.nwk

# Protein alignment (default)
FastTree alignment.fasta > tree.nwk

# With reproducible seed and GTR model
FastTree -gtr -nt -seed 12345 alignment.fasta > tree.nwk
```

See the full [CLI documentation](https://morgannprice.github.io/fasttree/) for all options.

## Library API

The library allows embedding FastTree in other programs without forking a subprocess or doing I/O through files.

### Quick Start

```c
#include "fasttree.h"

int main(void) {
    /* Configure */
    fasttree_config_t config;
    fasttree_config_init(&config);
    config.seq_type = FASTTREE_SEQ_NUCLEOTIDE;
    config.seed = 12345;

    /* Create context */
    fasttree_ctx_t *ctx = fasttree_create(&config);

    /* Build tree from in-memory alignment */
    const char *names[] = {"Seq1", "Seq2", "Seq3", "Seq4"};
    const char *seqs[]  = {
        "ACGTACGTACGTACGTACGT",
        "ACGTACGTACGTACGTACGA",
        "TGCATGCATGCATGCATGCA",
        "TGCATGCATGCATGCATGCG"
    };

    fasttree_tree_t *tree = NULL;
    fasttree_stats_t stats;
    int rc = fasttree_build(ctx, names, seqs, 4, 20, &tree, &stats);

    if (rc != FASTTREE_OK) {
        fprintf(stderr, "Error: %s\n", fasttree_last_error(ctx));
        fasttree_destroy(ctx);
        return 1;
    }

    /* Serialize to Newick */
    char *newick = fasttree_tree_to_newick(tree, /*show_support=*/1);
    printf("%s", newick);

    /* Cleanup */
    free(newick);
    fasttree_tree_free(tree);
    fasttree_destroy(ctx);
    return 0;
}
```

Compile with:
```bash
gcc -o my_program my_program.c -lfasttree -lm
# or link statically:
gcc -o my_program my_program.c libfasttree.a -lm
```

### API Reference

#### Lifecycle

| Function | Description |
|----------|-------------|
| `fasttree_config_init(config)` | Initialize config with defaults matching standard FastTree |
| `fasttree_create(config)` | Create a context. Returns NULL on error |
| `fasttree_build(ctx, names, seqs, nSeq, nPos, &tree, &stats)` | Build a tree from an in-memory alignment |
| `fasttree_destroy(ctx)` | Free a context and all memory it owns |

#### Tree Output

| Function | Description |
|----------|-------------|
| `fasttree_tree_to_newick(tree, show_support)` | Serialize to Newick string (caller frees with `free()`) |
| `fasttree_tree_free(tree)` | Free a tree returned by `fasttree_build` |

#### SOA (Structure of Arrays) Output

An alternative tree layout where each node field is a contiguous array. Better cache behavior for columnar bulk operations (rescaling branch lengths, filtering by support).

| Function | Description |
|----------|-------------|
| `fasttree_build_soa(ctx, names, seqs, nSeq, nPos, &tree, &stats)` | Build tree with SOA layout (single malloc) |
| `fasttree_tree_soa_to_newick(tree, show_support)` | Serialize SOA tree to Newick string |
| `fasttree_tree_soa_free(tree)` | Free a SOA tree |

```c
fasttree_tree_soa_t *tree = NULL;
fasttree_build_soa(ctx, names, seqs, nSeq, nPos, &tree, NULL);

/* Columnar access — iterate one field at a time */
for (int i = 0; i < tree->n_nodes; i++)
    total += tree->branch_length[i];

/* Children via offset into shared buffer */
for (int j = 0; j < tree->n_children[i]; j++) {
    int child = tree->_children_buf[tree->children_offset[i] + j];
}

fasttree_tree_soa_free(tree);  /* single free */
```

The entire SOA tree is one allocation — `fasttree_tree_soa_free` releases everything.

#### Error Handling

| Function | Description |
|----------|-------------|
| `fasttree_strerror(code)` | Human-readable string for an error code |
| `fasttree_last_error(ctx)` | Detailed error message from last failed operation |

Error codes: `FASTTREE_OK`, `FASTTREE_ERR_NOMEM`, `FASTTREE_ERR_INVALID_INPUT`, `FASTTREE_ERR_PARSE`, `FASTTREE_ERR_INTERNAL`, `FASTTREE_ERR_INVALID_CONFIG`, `FASTTREE_ERR_CANCELLED`.

### Configuration

All fields have sensible defaults set by `fasttree_config_init`. Common options:

```c
config.seq_type = FASTTREE_SEQ_NUCLEOTIDE;  // or FASTTREE_SEQ_PROTEIN (default)
config.model    = FASTTREE_MODEL_GTR;       // nucleotide: JC (default), GTR
                                            // protein: JTT (default), LG, WAG
config.n_bootstrap = 1000;     // SH-like support resamples (default 1000)
config.fastest     = 1;        // fastest heuristics, less accurate
config.slow        = 1;        // exhaustive search, more accurate
config.seed        = 12345;    // reproducible results
config.n_threads   = 4;        // OpenMP threads (0 = use OMP_NUM_THREADS)
```

### Thread Safety

Each `fasttree_ctx_t` is independent. It is safe to use different contexts concurrently from different threads. A single context must not be used from multiple threads simultaneously.

A context may be reused for multiple `fasttree_build` calls. The arena allocator is reset between builds automatically.

### Custom Allocator

```c
config.alloc_fn        = my_alloc;  // void *(*)(size_t, void*)
config.free_fn         = my_free;   // void (*)(void*, void*)
config.alloc_user_data = my_pool;   // passed to both callbacks
```

The library uses an arena allocator internally. `alloc_fn`/`free_fn` are called to allocate and free the arena's backing blocks (~1 MiB each). Must return 16-byte-aligned memory. If NULL (default), system `malloc`/`free` are used.

### Callbacks

```c
/* Log callback — receives verbose/debug messages */
void my_log(const char *msg, void *user_data) {
    fputs(msg, stderr);
}
config.log_callback = my_log;

/* Progress callback — return nonzero to cancel */
int my_progress(const char *stage, double frac_done, void *user_data) {
    printf("%s: %.0f%%\n", stage, frac_done * 100);
    return 0;  // 0 = continue, nonzero = cancel
}
config.progress_callback = my_progress;
```

### Known Limitations (v3.0)

- `FASTTREE_SEQ_AUTO` defaults to protein; auto-detection not yet implemented
- `config.start_newick` not yet supported (rejected at create time; use CLI `-intree` instead)
- `-constraints`, `-makematrix`, `-log`, `-trans` only available via CLI

## Algorithm

FastTree uses a five-stage pipeline:

1. **Heuristic Neighbor-Joining** with profiles and top-hits heuristic
2. **Minimum Evolution NNI+SPR** topology refinement
3. **Maximum-Likelihood NNI** with branch length optimization
4. **SH-like local support values** via resampling
5. **Gamma20 log-likelihood** rescaling (optional)

Substitution models: Jukes-Cantor or GTR (nucleotide), JTT, WAG, or LG (protein). Rate variation modeled with the CAT approximation.

For details, see Price, Dehal, and Arkin (2010) "FastTree 2 -- Approximately Maximum-Likelihood Trees for Large Alignments" *PLoS ONE* 5(3):e9490.

## License

GPL v2 or later. Copyright (C) 2008-2015 The Regents of the University of California. Original FastTree code by Morgan N. Price.
