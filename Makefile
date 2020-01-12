
CFLAGS ?= -Wall -O2 -march=native -fopenmp
LIBS ?= -ljpeg -lm

.PHONY: clean all

all: quantsmooth

clean:
	rm -rf quantsmooth

quantsmooth: quantsmooth.h idct.h

quantsmooth: quantsmooth.c
	$(CC) $(CFLAGS) -s -o $@ $< $(LIBS)

