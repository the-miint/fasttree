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

# API test (link statically to avoid LD_LIBRARY_PATH)
test_api: test_api.c libfasttree.a
	$(CC) $(CFLAGS) -o $@ $< libfasttree.a $(LDFLAGS)

# Ground truth test
test: FastTree FastTree.orig
	@echo "=== Ground truth tests ==="
	@for f in 16S.1 16S.2; do \
	  ./FastTree -seed 12345 -nt < testdata/16S500/$$f.p 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f" || echo "FAIL: $$f"; \
	done
	@for f in COG6 COG9; do \
	  ./FastTree -seed 12345 < testdata/BigCOGs/$$f.500.p 2>/dev/null | \
	    diff - testdata/ground_truth/$$f.nwk > /dev/null && \
	    echo "PASS: $$f" || echo "FAIL: $$f"; \
	done

PREFIX ?= /usr/local
install: libfasttree.a libfasttree.so FastTree fasttree.h
	install -d $(PREFIX)/lib $(PREFIX)/include $(PREFIX)/bin
	install -m 644 libfasttree.a $(PREFIX)/lib/
	install -m 755 libfasttree.so $(PREFIX)/lib/
	install -m 644 fasttree.h $(PREFIX)/include/
	install -m 755 FastTree $(PREFIX)/bin/

clean:
	rm -f *.o *.pic.o libfasttree.a libfasttree.so FastTree FastTree.orig test_api

.PHONY: all clean install test
