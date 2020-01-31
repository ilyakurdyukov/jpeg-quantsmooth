
APPNAME ?= jpegqs
SRCNAME ?= quantsmooth.c
# machine flags
MFLAGS := -march=native
# multithreading options
MTOPTS := -fopenmp

CFLAGS_LIB := -Wall -O2 $(MFLAGS)
CFLAGS := $(CFLAGS_LIB) -Wextra -pedantic $(MTOPTS)

.PHONY: clean all

all: $(APPNAME)

$(APPNAME): quantsmooth.h idct.h

ifeq ($(JPEGSRC),)
JPEGLIB = -ljpeg

clean:
	rm -f $(APPNAME)

$(APPNAME): $(SRCNAME)
	$(CC) -DAPPNAME=$(APPNAME) $(CFLAGS) -s -o $@ $< $(JPEGLIB) -lm
else
OBJDIR ?= $(JPEGSRC)
ALLSRC := $(patsubst $(JPEGSRC)/%.c,%,$(wildcard $(JPEGSRC)/*.c))
SOURCES := jutils jmemmgr jmemnobs jcomapi jerror \
  jdapimin jdcoefct jdmarker jdhuff jdinput jdtrans \
  jcapimin jcmaster jcmarker jchuff jcparam jctrans \
	rdswitch cdjpeg transupp jdatasrc jdatadst jidctint
# version specific sources
SOURCES += $(filter jdphuff jcphuff jaricom jdarith jcarith,$(ALLSRC))

OBJLIST := $(patsubst %,$(OBJDIR)/%.o,$(SOURCES))

clean:
	rm -f $(APPNAME) $(OBJLIST)

$(OBJDIR)/%.o: $(JPEGSRC)/%.c
	$(CC) $(CFLAGS_LIB) -I$(JPEGSRC) -I. -c -o $@ $<

$(APPNAME): $(SRCNAME) $(OBJLIST)
	$(CC) $(CFLAGS) -DAPPNAME=$(APPNAME) -I$(JPEGSRC) -I. -s -o $@ $< $(OBJLIST) -lm
endif

