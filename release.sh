#!/bin/sh

l="-ljpeg -lm"
f="-fopenmp -static"
jpeg="jpeg-6b"
[ -d $jpeg ] && l="-I $jpeg $jpeg/libjpeg.a -lm"

make LIBS="$l" CFLAGS="$f -O2 -mavx2" clean all && cp quantsmooth quantsmooth_avx2
make LIBS="$l" CFLAGS="$f -O2 -msse2" clean all && cp quantsmooth quantsmooth_sse2
make LIBS="$l" CFLAGS="$f -O3 -mno-sse2" clean all

