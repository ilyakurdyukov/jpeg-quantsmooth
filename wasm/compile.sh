#!/bin/bash

source ../emsdk/emsdk_env.sh --build=Release

debug=0
# --pre-js shell.js
emflags="-O3 -g0 --closure 1"
[ $debug == 1 ] && emflags="-O2 -g1 -s ASSERTIONS=1"

jpeg="jpeg-6b"

CFLAGS_LIB="$emflags -Wno-shift-negative-value"
CFLAGS_APP="-ffast-math -DWASM -DNO_SIMD $emflags --shell-file shell.html -s EXPORTED_FUNCTIONS=\"['_malloc', '_free']\" -s ALLOW_MEMORY_GROWTH=1 -DWITH_JPEGSRC -I$jpeg -I."

make JPEGSRC="$jpeg" CC="emcc" CFLAGS_LIB="$CFLAGS_LIB" CFLAGS_APP="$CFLAGS_APP" APPNAME="quantsmooth.html" clean all
test -f quantsmooth.html && mv quantsmooth.html index.html

echo "press enter..."; read dummy < /dev/tty

