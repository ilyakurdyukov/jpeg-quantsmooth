#!/bin/sh

l="-ljpeg -lm"
f="-O3 -fopenmp -static"
jpeg="jpeg-6b"
[ -d $jpeg ] && l="-I $jpeg $jpeg/libjpeg.a -lm"

make LIBS="$l" CFLAGS="$f -mavx2" clean all && cp quantsmooth quantsmooth_avx2
make LIBS="$l" CFLAGS="$f -msse4" clean all && cp quantsmooth quantsmooth_sse4
make LIBS="$l" CFLAGS="$f" clean all

