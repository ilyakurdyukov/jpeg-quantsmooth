
LIBNAME ?= jpegqs.dll
SONAME ?= JPEGQS.DLL
MTOPTS ?= -fopenmp
CFLAGS := -O2 -Wall -Wextra -pedantic -municode -I.. $(MTOPTS)
JPEGLIB ?= -ljpeg -static

.PHONY: all clean

all: $(LIBNAME)

clean:
	rm -f $(LIBNAME)

$(LIBNAME): plugin.c
	$(CC) $(CFLAGS) -shared -Wl,-soname,$(SONAME) -o $@ $< $(JPEGLIB) -Wl,--version-script,plugin-vers.txt -Wl,--gc-sections
	strip $@

