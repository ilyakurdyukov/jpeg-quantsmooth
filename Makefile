
APPNAME ?= quantsmooth
CFLAGS ?= -Wall -O2 -march=native -fopenmp
LIBS ?= -ljpeg -lm

.PHONY: clean all

all: $(APPNAME)

clean:
	rm -rf $(APPNAME)

$(APPNAME): quantsmooth.h idct.h

$(APPNAME): quantsmooth.c
	$(CC) -DAPPNAME=$(APPNAME) $(CFLAGS) -s -o $@ $< $(LIBS)

