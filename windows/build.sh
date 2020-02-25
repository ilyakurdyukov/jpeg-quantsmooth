#!/bin/sh
bits=${1:-""}

test -f ../ldscript$bits.txt && link="-Wl,-T,../ldscript$bits.txt" || link=

make APPNAME="jpegqs${bits}_gui" JPEGLIB="../libjpegqs${bits}.a $lib" LFLAGS="$link" LIBNAME="$name" clean all
