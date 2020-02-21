
SRCDEPS := quantsmooth.h idct.h libjpegqs.h
SRCNAME ?= quantsmooth.c
ifeq ($(SRCNAME),jpegqs-mini.c)
APPNAME ?= jpegqs-mini
$(APPNAME): Makefile
else
ifeq ($(SRCNAME),example.c)
APPNAME ?= example
else
APPNAME ?= jpegqs
endif
$(APPNAME): Makefile $(SRCDEPS)
endif
SIMD ?= native
# machine flags
MFLAGS ?= 
SIMDFLG :=
SIMDOBJ :=
ifeq ($(SIMD),select)
SIMDOBJ := jpegqs_avx2.o jpegqs_sse2.o jpegqs_base.o
else ifeq ($(SIMD),none)
SIMDFLG := -DNO_SIMD
else ifeq ($(SIMD),native)
SIMDFLG := -march=native
else ifeq ($(SIMD),avx2)
SIMDFLG := -mavx2 -mfma
else ifeq ($(SIMD),sse2)
SIMDFLG := -msse2
endif
# multithreading options
MTOPTS := -fopenmp

CFLAGS_LIB := -Wall -O2 $(MFLAGS) $(SIMDFLG)
CFLAGS_APP := $(CFLAGS_LIB) -Wextra -pedantic $(MTOPTS)
ifeq ($(SIMD),select)
CFLAGS_APP += -DSIMD_SELECT
endif

.PHONY: clean all app lib

app: $(APPNAME)
all: app lib
lib: lib$(APPNAME).a

WGET_CMD = @echo "run make with WGET_CMD=wget to allow file downloads" ; echo "DISABLED:" wget

jpegsrc.v%.tar.gz: 
	$(WGET_CMD) -O $@ "https://www.ijg.org/files/$@"
	test -f $@
jpeg-%/jutils.c: jpegsrc.v%.tar.gz
	tar -xzf jpegsrc.v$(patsubst jpeg-%/jutils.c,%,$@).tar.gz
	touch $@
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
	touch $@
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
CFLAGS_APP += $(filter -I%,$(JPEGLIB))

clean:
	rm -f $(APPNAME) jpegqs_*.o libjpegqs*.o lib$(APPNAME).a

$(APPNAME): $(SRCNAME) $(SIMDOBJ)
	$(CC) $(CFLAGS_APP) -DAPPNAME=$(APPNAME) -s -o $@ $< -Wl,--gc-sections $(JPEGLIB) $(SIMDOBJ) -lm
else
OBJDIR ?= $(JPEGSRC)
ALLSRC := $(patsubst $(JPEGSRC)/%.c,%,$(wildcard $(JPEGSRC)/*.c))
SOURCES := jutils jmemmgr jmemnobs jcomapi jerror \
  jdapimin jdcoefct jdmarker jdhuff jdinput jdtrans \
  jcapimin jcmaster jcmarker jchuff jcparam jctrans \
	rdswitch cdjpeg transupp jdatasrc jdatadst
ifeq ($(SRCNAME),jpegqs-mini.c)
SOURCES += jidctint jfdctint
else ifeq ($(SRCNAME),example.c)
SOURCES += jidctint jidctfst jidctflt jquant1 jquant2 \
	jdapistd jdmaster jdcolor jdpostct jddctmgr jdsample jdmerge jdmainct
SOURCES += $(filter jidctred,$(ALLSRC))
endif
# version specific sources
SOURCES += $(filter jdphuff jcphuff jaricom jdarith jcarith,$(ALLSRC))

OBJLIST := $(patsubst %,$(OBJDIR)/%.o,$(SOURCES))
CFLAGS_APP += -DWITH_JPEGSRC -I$(JPEGSRC) -I.

clean:
	rm -f $(APPNAME) $(OBJLIST) jpegqs_*.o libjpegqs*.o lib$(APPNAME).a

$(OBJDIR)/%.o: $(JPEGSRC)/%.c
	$(CC) $(CFLAGS_LIB) -I$(JPEGSRC) -I. -c -o $@ $<

$(APPNAME): $(SRCNAME) $(OBJLIST) $(SIMDOBJ)
	$(CC) $(CFLAGS_APP) -DAPPNAME=$(APPNAME) -s -o $@ $< -Wl,--gc-sections $(OBJLIST) $(SIMDOBJ) -lm
endif

ifeq ($(SRCNAME),example.c)
SIMDSEL_FLAGS ?=
else
SIMDSEL_FLAGS ?= -DTRANSCODE_ONLY -DWITH_LOG
endif

jpegqs_avx2.o: libjpegqs.c $(SRCDEPS)
	$(CC) $(SIMDSEL_FLAGS) -DSIMD_NAME=avx2 -mavx2 -mfma $(CFLAGS_APP) -DNO_HELPERS -c -o $@ $<
jpegqs_sse2.o: libjpegqs.c $(SRCDEPS)
	$(CC) $(SIMDSEL_FLAGS) -DSIMD_NAME=sse2 -msse2 $(CFLAGS_APP) -DNO_HELPERS -c -o $@ $<
jpegqs_base.o: libjpegqs.c $(SRCDEPS)
	$(CC) $(SIMDSEL_FLAGS) -DSIMD_NAME=base $(CFLAGS_APP) -c -o $@ $<

ifeq ($(SIMD),select)
lib$(APPNAME).a: libjpegqs_avx2.o libjpegqs_sse2.o libjpegqs_base.o
endif
lib$(APPNAME).a: libjpegqs.o
	$(AR) -rsc $@ $^

libjpegqs.o: libjpegqs.c $(SRCDEPS)
	$(CC) $(CFLAGS_APP) -c -o $@ $<
libjpegqs_avx2.o: libjpegqs.c $(SRCDEPS)
	$(CC) -DSIMD_NAME=avx2 -mavx2 -mfma $(CFLAGS_APP) -DNO_HELPERS -c -o $@ $<
libjpegqs_sse2.o: libjpegqs.c $(SRCDEPS)
	$(CC) -DSIMD_NAME=sse2 -msse2 $(CFLAGS_APP) -DNO_HELPERS -c -o $@ $<
libjpegqs_base.o: libjpegqs.c $(SRCDEPS)
	$(CC) -DSIMD_NAME=base $(CFLAGS_APP) -c -o $@ $<

