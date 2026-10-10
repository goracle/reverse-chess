#!/bin/sh
# Build the Rust move generator and drop the shared library next to engine.py.
set -e
cd "$(dirname "$0")/native"
cargo build --release
case "$(uname -s)" in
  Darwin) lib=librc_engine.dylib ;;
  MINGW*|MSYS*|CYGWIN*) lib=rc_engine.dll ;;
  *) lib=librc_engine.so ;;
esac
cp "target/release/$lib" "../$lib"
echo "built $(cd .. && pwd)/$lib"
