#!/bin/bash
nvcc -arch=sm_13 -O -Icommon/inc $1.cu -Llib -lcutil_x86_64 -o $1 
rm .ref.swp .out.swp
cuda-memcheck ./$1 -i $2.pgm -o $2Out.pgm
#./$1 -i $2.pgm -o $2Out.pgm
hexdump $2Out.pgm > out
hexdump reference.pgm > ref
./testDiffs -v reference.pgm $2Out.pgm
vim -d ref out
