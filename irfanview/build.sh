#!/bin/sh
jpeg=${1:-"jpeg-6b"}
bits=${2:-""}

lib="-ljpeg -static"
[ -d $jpeg ] && lib="-I$jpeg $jpeg/libjpeg.a -static"
name="JpegQS.dll"
[ "$bits" ] && {
	mkdir -p x"$bits"
	name="x"$bits"/$name"
}

test -d ../winlib$bits && lib="$lib -L../winlib$bits"
test -f ../ldscript$bits.txt && lib="$lib -Wl,-T,../ldscript$bits.txt"

make JPEGLIB="../libjpegqs${bits}.a $lib" LIBNAME="$name" clean all
