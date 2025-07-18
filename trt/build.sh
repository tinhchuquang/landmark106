#!/bin/bash
set -e
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
echo "Done. Run with: ./build/pfld"