#!/bin/bash

source ../emsdk/emsdk_env.sh --build=Release

jpeg="jpeg-6b"
cp jconfig.h $jpeg
list="$jpeg/jutils.c $jpeg/jmemmgr.c $jpeg/jmemnobs.c $jpeg/jcomapi.c $jpeg/jerror.c"
list="$list $jpeg/jdapimin.c $jpeg/jdcoefct.c $jpeg/jdmarker.c $jpeg/jdhuff.c $jpeg/jdphuff.c $jpeg/jdinput.c $jpeg/jdtrans.c"
list="$list $jpeg/jcapimin.c $jpeg/jcmaster.c $jpeg/jcmarker.c $jpeg/jchuff.c $jpeg/jcphuff.c $jpeg/jcparam.c $jpeg/jctrans.c"

debug=0
# --pre-js shell.js
emflags="-O3 -g0 --closure 1"
[ $debug == 1 ] && emflags="-O2 -g1 -s ASSERTIONS=1"

emcc -I $jpeg ../quantsmooth.c $list -o quantsmooth.html --shell-file shell.html \
	$emflags -Wno-shift-negative-value \
	-s EXPORTED_FUNCTIONS="['_malloc', '_free']" -s ALLOW_MEMORY_GROWTH=1
mv quantsmooth.html index.html

echo "press enter..."; read dummy < /dev/tty

