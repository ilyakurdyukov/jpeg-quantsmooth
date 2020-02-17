#!/bin/sh
jpeg=${1:-"jpeg-6b"}
bits=${2:-""}

lib="-ljpeg -static"
[ -d $jpeg ] && lib="-DWITH_JPEGSRC -I$jpeg $jpeg/libjpeg.a -static"

make JPEGLIB="$lib" MFLAGS="-O2 -mavx2 -municode" APPNAME="jpegqs${bits}_avx2" clean all
make JPEGLIB="$lib" MFLAGS="-O2 -msse2 -municode" APPNAME="jpegqs${bits}_sse2" clean all
make JPEGLIB="$lib" MFLAGS="-O3 -DNO_SIMD -municode" APPNAME="jpegqs${bits}" clean all

