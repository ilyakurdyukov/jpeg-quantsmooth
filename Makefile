
APPNAME ?= jpegqs
SRCNAME ?= quantsmooth.c
# machine flags
MFLAGS := -march=native
# multithreading options
MTOPTS := -fopenmp

ifneq ($(SRCNAME),jpegqs-mini.c)
$(APPNAME): quantsmooth.h idct.h
endif

CFLAGS_LIB := -Wall -O2 $(MFLAGS)
CFLAGS_APP := $(CFLAGS_LIB) -Wextra -pedantic $(MTOPTS)

.PHONY: clean all

all: $(APPNAME)

WGET_CMD = @echo "run make with WGET_CMD=wget to allow file downloads" ; echo "DISABLED:" wget

jpegsrc.v%.tar.gz: 
	$(WGET_CMD) -O $@ "https://www.ijg.org/files/$@"
	test -f $@
jpeg-%/jutils.c: jpegsrc.v%.tar.gz
	tar -xzf jpegsrc.v$(patsubst jpeg-%/jutils.c,%,$@).tar.gz
jpeg-%/Makefile: jpeg-%/jutils.c
	cd $(patsubst %/Makefile,%,$@) && ./configure
jpeg-%/libjpeg.a: jpeg-%/Makefile
	cd $(patsubst %/libjpeg.a,%,$@) && $(MAKE) all && test -d .libs && cp .libs/libjpeg.a . || true
.PRECIOUS: jpegsrc.v%.tar.gz jpeg-%/jutils.c jpeg-%/Makefile

libjpeg-turbo-%.tar.gz:
	$(WGET_CMD) -O $@ "https://sourceforge.net/projects/libjpeg-turbo/files/$(patsubst libjpeg-turbo-%.tar.gz,%,$@)/libjpeg-turbo-$(patsubst libjpeg-turbo-%.tar.gz,%,$@).tar.gz"
	test -f $@
libjpeg-turbo-%/jutils.c: libjpeg-turbo-%.tar.gz
	tar -xzf $(patsubst %/jutils.c,%,$@).tar.gz
.PRECIOUS: libjpeg-turbo-%.tar.gz libjpeg-turbo-%/jutils.c
libjpeg-turbo-1.%/Makefile: libjpeg-turbo-1.%/jutils.c
	cd $(patsubst %/Makefile,%,$@) && ./configure
libjpeg-turbo-1.%/libjpeg.a: libjpeg-turbo-1.%/Makefile
	cd $(patsubst %/libjpeg.a,%,$@) && $(MAKE) all && cp .libs/lib*jpeg.a .
.PRECIOUS: libjpeg-turbo-1.%/Makefile
libjpeg-turbo-2.%/.libs/Makefile: libjpeg-turbo-2.%/jutils.c
	mkdir -p $(patsubst %/Makefile,%,$@)
	cd $(patsubst %/Makefile,%,$@) && cmake -G"Unix Makefiles" ..
libjpeg-turbo-2.%/libjpeg.a: libjpeg-turbo-2.%/.libs/Makefile
	cd $(patsubst %/Makefile,%,$<) && $(MAKE) all && cp jconfig*.h lib*jpeg.a ..
.PRECIOUS: libjpeg-turbo-2.%/.libs/Makefile

ifeq ($(JPEGSRC),)
JPEGLIB = -ljpeg

clean:
	rm -f $(APPNAME)

$(APPNAME): $(SRCNAME)
	$(CC) -DAPPNAME=$(APPNAME) $(CFLAGS_APP) -s -o $@ $< $(JPEGLIB) -lm
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
	$(CC) $(CFLAGS_APP) -DAPPNAME=$(APPNAME) -DWITH_JPEGSRC -I$(JPEGSRC) -I. -s -o $@ $< $(OBJLIST) -lm
endif

