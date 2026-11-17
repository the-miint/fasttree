CC      ?= gcc
CFLAGS  = -Wall -O3 -finline-functions -funroll-loops -DUSE_DOUBLE
LDFLAGS = -lm

# Shared library needs position-independent code
CFLAGS_PIC = $(CFLAGS) -fPIC

# The wasm32 build takes CFLAGS before OMP=1 adds OpenMP: it has none.
WASM_CFLAGS := $(CFLAGS)

# Optional OpenMP support: make OMP=1
ifdef OMP
  CFLAGS     += -DOPENMP -fopenmp
  CFLAGS_PIC += -DOPENMP -fopenmp
  LDFLAGS    += -fopenmp
endif

HEADERS = fasttree.h fasttree_internal.h

# --- Build targets ---

all: FastTree libfasttree.a libfasttree.so

# Original binary (sanity check)
FastTree.orig: FastTree.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Library object files (no main)
fasttree_core.o: fasttree_core.c $(HEADERS)
	$(CC) $(CFLAGS) -DFASTTREE_NO_MAIN -c -o $@ $<

fasttree_api.o: fasttree_api.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Static library
libfasttree.a: fasttree_core.o fasttree_api.o
	ar rcs $@ $^

# PIC objects for shared library
fasttree_core.pic.o: fasttree_core.c $(HEADERS)
	$(CC) $(CFLAGS_PIC) -DFASTTREE_NO_MAIN -c -o $@ $<

fasttree_api.pic.o: fasttree_api.c $(HEADERS)
	$(CC) $(CFLAGS_PIC) -c -o $@ $<

# Shared library
libfasttree.so: fasttree_core.pic.o fasttree_api.pic.o
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# CLI binary (uses main from fasttree_core.c + API)
FastTree: fasttree_core.c fasttree_api.o $(HEADERS)
	$(CC) $(CFLAGS) -o $@ fasttree_core.c fasttree_api.o $(LDFLAGS)

# OMP-forced CLI binary: always compiled with -DOPENMP -fopenmp, regardless of
# the OMP=1 flag. Used by the single-thread OMP parity regression test, which
# guards against the upstream MLQuartetNNI divergence where #ifdef OPENMP
# unconditionally disables the star-topology shortcut and replaces the
# bStarTest early-exit with an omp section directive.
FastTree.omp: fasttree_core.c fasttree_api.c $(HEADERS)
	$(CC) $(CFLAGS) -DOPENMP -fopenmp -o $@ fasttree_core.c fasttree_api.c -lm -fopenmp

# API test (link statically to avoid LD_LIBRARY_PATH)
test_api: test_api.c libfasttree.a
	$(CC) $(CFLAGS) -o $@ $< libfasttree.a $(LDFLAGS)

# Library parity driver: reads PHYLIP, calls fasttree_build_soa, writes Newick.
# Used by the parity test to confirm the library API path is bit-equal to FastTree.orig.
test_parity: test_parity.c libfasttree.a
	$(CC) $(CFLAGS) -o $@ $< libfasttree.a $(LDFLAGS)

# Thread-safety driver: spawns N threads, each runs fasttree_build_soa with its
# own ctx on the same input, asserts every thread's Newick equals the ground truth.
test_threads: test_threads.c libfasttree.a
	$(CC) $(CFLAGS) -pthread -o $@ $< libfasttree.a $(LDFLAGS)

# --- Tree checks ---
#
# $(call check_trees,<label>,<command>,<input prefix>) runs <command> on the
# four saved inputs and checks each with check_tree.sh: the command must
# exit 0 and print the saved reference tree. <input prefix> is "<" for a
# command that reads the alignment on standard input, empty for one that
# takes its path as an argument.
check_trees = st=0; \
  for f in 16S.1 16S.2; do \
    sh ./check_tree.sh "$$f ($(1))" testdata/ground_truth/$$f.nwk \
      "$(2) $(3) testdata/16S500/$$f.p -nt" || st=1; \
  done; \
  for f in COG6 COG9; do \
    sh ./check_tree.sh "$$f ($(1))" testdata/ground_truth/$$f.nwk \
      "$(2) $(3) testdata/BigCOGs/$$f.500.p" || st=1; \
  done; \
  exit $$st

# --- wasm32 (emscripten), run under node ---
#
# Objects go to build-wasm/ so they never mix with the native ones.
# Assertions stay on, as in the native build. NODERAWFS lets test_parity
# read its PHYLIP input from the host filesystem; the 8 MB stack replaces
# emscripten's 64 KB default.
EMCC         ?= emcc
WASM_DIR      = build-wasm
WASM_LDFLAGS  = -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=8MB -sNODERAWFS=1 -sEXIT_RUNTIME=1
WASM_LIB      = $(WASM_DIR)/fasttree_core.o $(WASM_DIR)/fasttree_api.o

$(WASM_DIR)/fasttree_core.o: fasttree_core.c $(HEADERS)
	@mkdir -p $(WASM_DIR)
	$(EMCC) $(WASM_CFLAGS) -DFASTTREE_NO_MAIN -c -o $@ $<

$(WASM_DIR)/fasttree_api.o: fasttree_api.c $(HEADERS)
	@mkdir -p $(WASM_DIR)
	$(EMCC) $(WASM_CFLAGS) -c -o $@ $<

$(WASM_DIR)/test_api.js: test_api.c $(WASM_LIB)
	$(EMCC) $(WASM_CFLAGS) -o $@ $< $(WASM_LIB) $(WASM_LDFLAGS)

$(WASM_DIR)/test_parity.js: test_parity.c $(WASM_LIB)
	$(EMCC) $(WASM_CFLAGS) -o $@ $< $(WASM_LIB) $(WASM_LDFLAGS)

# The API tests, then the library parity tests against the same saved
# references as `make test`.
test-wasm: $(WASM_DIR)/test_api.js $(WASM_DIR)/test_parity.js
	@echo "=== API smoke tests (wasm32, node) ==="
	@node $(WASM_DIR)/test_api.js
	@echo "=== Library parity tests (wasm32, node, API vs saved reference) ==="
	@$(call check_trees,wasm library,node $(WASM_DIR)/test_parity.js,)

# Ground truth tests use FastTree.orig (always non-OMP, when invoked as
# `make test`) for the saved reference. The library parity tests confirm
# fasttree_build_soa output is bit-identical on the same inputs.
# The OMP single-thread parity tests use a dedicated FastTree.omp binary
# (always built with -DOPENMP) to guarantee the OMP code path is exercised
# at one thread, regression-guarding the MLQuartetNNI single-thread fix.
test: FastTree.orig FastTree.omp test_parity test_threads
	@echo "=== Ground truth tests (CLI vs saved reference) ==="
	@$(call check_trees,cli,./FastTree.orig -seed 12345,<)
	@echo "=== Library parity tests (API vs saved reference) ==="
	@$(call check_trees,library,./test_parity,)
	@echo "=== OMP single-thread parity tests (regression for upstream OPENMP override) ==="
	@$(call check_trees,omp 1 thread,OMP_NUM_THREADS=1 ./FastTree.omp -seed 12345,<)
	@echo "=== Thread-safety tests (concurrent contexts vs reference) ==="
	@./test_threads testdata/16S500/16S.1.p testdata/ground_truth/16S.1.nwk 4 -nt 2>/dev/null \
	  && echo "PASS: 16S.1 (4 threads)" || { echo "FAIL: 16S.1 (4 threads)"; exit 1; }
	@./test_threads testdata/BigCOGs/COG6.500.p testdata/ground_truth/COG6.nwk 4 2>/dev/null \
	  && echo "PASS: COG6 (4 threads)" || { echo "FAIL: COG6 (4 threads)"; exit 1; }

PREFIX ?= /usr/local
install: libfasttree.a libfasttree.so FastTree fasttree.h
	install -d $(PREFIX)/lib $(PREFIX)/include $(PREFIX)/bin
	install -m 644 libfasttree.a $(PREFIX)/lib/
	install -m 755 libfasttree.so $(PREFIX)/lib/
	install -m 644 fasttree.h $(PREFIX)/include/
	install -m 755 FastTree $(PREFIX)/bin/

clean:
	rm -f *.o *.pic.o libfasttree.a libfasttree.so FastTree FastTree.omp FastTree.orig test_api test_parity test_threads
	rm -rf $(WASM_DIR) test-out

.PHONY: all clean install test test-wasm
