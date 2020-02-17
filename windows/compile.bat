@echo off

set OLDPATH=%PATH%

set PATH=C:\msys64\usr\bin;C:\msys64\mingw32\bin;%OLDPATH%
C:\msys64\usr\bin\sh -c "make clean all"

set PATH=C:\msys64\usr\bin;C:\msys64\mingw64\bin;%OLDPATH%
C:\msys64\usr\bin\sh -c "make APPNAME=jpegqs64_gui clean all"

pause
