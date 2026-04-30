CC      ?= gcc
CFLAGS  = -Wall -O3 -finline-functions -funroll-loops -DUSE_DOUBLE
LDFLAGS = -lm

# Shared library needs position-independent code
CFLAGS_PIC = $(CFLAGS) -fPIC

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

# Ground truth tests use FastTree.orig (always non-OMP, when invoked as
# `make test`) for the saved reference. The library parity tests confirm
# fasttree_build_soa output is bit-identical on the same inputs.
# The OMP single-thread parity tests use a dedicated FastTree.omp binary
# (always built with -DOPENMP) to guarantee the OMP code path is exercised
# at one thread, regression-guarding the MLQuartetNNI single-thread fix.
test: FastTree.orig FastTree.omp test_parity test_threads
	@echo "=== Ground truth tests (CLI vs saved reference) ==="
	@for f in 16S.1 16S.2; do \
	  ./FastTree.orig -seed 12345 -nt < testdata/16S500/$$f.p 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f" || echo "FAIL: $$f"; \
	done
	@for f in COG6 COG9; do \
	  ./FastTree.orig -seed 12345 < testdata/BigCOGs/$$f.500.p 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f" || echo "FAIL: $$f"; \
	done
	@echo "=== Library parity tests (API vs saved reference) ==="
	@for f in 16S.1 16S.2; do \
	  ./test_parity testdata/16S500/$$f.p -nt 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f (library)" || echo "FAIL: $$f (library)"; \
	done
	@for f in COG6 COG9; do \
	  ./test_parity testdata/BigCOGs/$$f.500.p 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f (library)" || echo "FAIL: $$f (library)"; \
	done
	@echo "=== OMP single-thread parity tests (regression for upstream OPENMP override) ==="
	@for f in 16S.1 16S.2; do \
	  OMP_NUM_THREADS=1 ./FastTree.omp -seed 12345 -nt < testdata/16S500/$$f.p 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f (omp 1 thread)" || echo "FAIL: $$f (omp 1 thread)"; \
	done
	@for f in COG6 COG9; do \
	  OMP_NUM_THREADS=1 ./FastTree.omp -seed 12345 < testdata/BigCOGs/$$f.500.p 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f (omp 1 thread)" || echo "FAIL: $$f (omp 1 thread)"; \
	done
	@echo "=== Thread-safety tests (concurrent contexts vs reference) ==="
	@./test_threads testdata/16S500/16S.1.p testdata/ground_truth/16S.1.nwk 4 -nt 2>/dev/null \
	  && echo "PASS: 16S.1 (4 threads)" || echo "FAIL: 16S.1 (4 threads)"
	@./test_threads testdata/BigCOGs/COG6.500.p testdata/ground_truth/COG6.nwk 4 2>/dev/null \
	  && echo "PASS: COG6 (4 threads)" || echo "FAIL: COG6 (4 threads)"

PREFIX ?= /usr/local
install: libfasttree.a libfasttree.so FastTree fasttree.h
	install -d $(PREFIX)/lib $(PREFIX)/include $(PREFIX)/bin
	install -m 644 libfasttree.a $(PREFIX)/lib/
	install -m 755 libfasttree.so $(PREFIX)/lib/
	install -m 644 fasttree.h $(PREFIX)/include/
	install -m 755 FastTree $(PREFIX)/bin/

clean:
	rm -f *.o *.pic.o libfasttree.a libfasttree.so FastTree FastTree.omp FastTree.orig test_api test_parity test_threads

.PHONY: all clean install test
