#!/bin/sh
# Portable Wine + 32-bit Windows Python for the NINR engine (SDR#'s shark.dll is x86 Windows).
# Nothing is installed system-wide; delete tools/wine and tools/winpython to remove it.
# You also need your own copy of SDR# ("next" build): export SORB_SDRSHARP=/path/to/sdrsharp-x86-next
set -eu
cd "$(dirname "$0")"
if [ ! -d wine ]; then
  mkdir -p wine && cd wine
  curl -fL -o wine.tar.xz https://github.com/Gcenx/macOS_Wine_builds/releases/download/11.16/wine-staging-11.16-osx64.tar.xz
  echo "cd68f230c773a761b8a0423a08c51fbe49e89b6e52246f3866898d259a4988c6  wine.tar.xz" | shasum -a 256 -c
  tar xf wine.tar.xz && rm wine.tar.xz && cd ..
fi
if [ ! -d winpython ]; then
  mkdir -p winpython && cd winpython
  curl -fL -o py.zip https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-win32.zip
  unzip -q py.zip && rm py.zip && cd ..
fi
echo "NINR toolchain ready (Wine 11.16 staging, Python 3.12.10 win32)."
