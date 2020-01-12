#!/bin/sh

jpeg="jpeg-6b"
[ -z "$1" ] || jpeg="$1"
bits=""
[ -z "$2" ] || bits="$2"

l="-ljpeg -lm"
f="-fopenmp -static"
[ -d $jpeg ] && l="-I $jpeg $jpeg/libjpeg.a -lm"

make LIBS="$l" CFLAGS="$f -O2 -mavx2" clean all && cp quantsmooth quantsmooth${bits}_avx2
make LIBS="$l" CFLAGS="$f -O2 -msse2" clean all && cp quantsmooth quantsmooth${bits}_sse2
make LIBS="$l" CFLAGS="$f -O3 -mno-sse2" clean all && cp quantsmooth quantsmooth${bits}

