#!/bin/bash

source ../emsdk/emsdk_env.sh --build=Release

debug=0
async=1
simd=0
# --pre-js shell.js
emflags="-O3 -g0 --closure 1"
[ $debug -ne 0 ] && emflags="-O2 -g1 -s ASSERTIONS=1"

jpeg="jpeg-6b"

CFLAGS_LIB="$emflags -Wno-shift-negative-value"
CFLAGS_APP="-ffast-math -DWASM -DNO_SIMD $emflags -DWITH_JPEGSRC -I$jpeg -I."
LFLAGS="--shell-file shell.html -s EXPORTED_FUNCTIONS=\"['_malloc', '_free']\" -s ALLOW_MEMORY_GROWTH=1"

[ $async -ne 0 ] && {
	CFLAGS_APP="$CFLAGS_APP -DWASM_ASYNC -s ASYNCIFY=1 -s ASYNCIFY_IGNORE_INDIRECT"
}

[ $simd -ge 1 ] && CFLAGS_APP="$CFLAGS_APP -msimd128"
[ $simd -ge 2 ] && CFLAGS_APP="$CFLAGS_APP -msse2"

make JPEGSRC="$jpeg" CC="emcc" CFLAGS_LIB="$CFLAGS_LIB" CFLAGS_APP="$CFLAGS_APP" LFLAGS="$LFLAGS" APPNAME="quantsmooth.html" clean app
test -f quantsmooth.html && mv quantsmooth.html index.html

echo "press enter..."; read dummy < /dev/tty

