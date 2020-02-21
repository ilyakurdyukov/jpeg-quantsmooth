#!/bin/sh
jpeg=${1:-"jpeg-6b"}
bits=${2:-""}

lib="-ljpeg -static"
[ -d $jpeg ] && lib="-DWITH_JPEGSRC -I$jpeg $jpeg/libjpeg.a -static"

# WinXP GetTickCount64 fix
[ -d winlib$bits ] && lib="$lib -Lwinlib$bits"

# make JPEGLIB="$lib" SIMD=avx2 MFLAGS="-municode" APPNAME="jpegqs${bits}_avx2" clean app
# make JPEGLIB="$lib" SIMD=sse2 MFLAGS="-municode" APPNAME="jpegqs${bits}_sse2" clean app
# make JPEGLIB="$lib" SIMD=none MFLAGS="-O3 -municode" APPNAME="jpegqs${bits}_none" clean app

make JPEGLIB="$lib" SIMD=select MFLAGS="-municode -ffast-math" APPNAME="jpegqs${bits}" clean all

