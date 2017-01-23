#!/bin/bash
nvcc -arch=sm_13 -O -Icommon/inc $1.cu -Llib -lcutil_x86_64 -o $1 
#rm .ref.swp .out.swp

# LENA
echo "\nImage: lena.pgm"
#cuda-memcheck ./$1 -i lena.pgm -o lenaOut.pgm
./$1 -i lena.pgm -o lenaOut.pgm
#./$1 -i $2.pgm -o $2Out.pgm
hexdump lenaOut.pgm > out
hexdump reference.pgm > ref
./testDiffs reference.pgm lenaOut.pgm
#vim -d ref out

# HOUSE
echo "\nImage: house.pgm"
#cuda-memcheck ./$1 -i house.pgm -o houseOut.pgm
./$1 -i house.pgm -o houseOut.pgm
hexdump houseOut.pgm > out
hexdump reference.pgm > ref
./testDiffs reference.pgm houseOut.pgm
#vim -d ref out

# CHESS
echo "\nImage: chessRotate1.pgm"
#cuda-memcheck ./$1 -i chessRotate1.pgm -o chessRotate1Out.pgm
./$1 -i chessRotate1.pgm -o chessRotate1Out.pgm
hexdump chessRotate1Out.pgm > out
hexdump reference.pgm > ref
./testDiffs reference.pgm chessRotate1Out.pgm
#vim -d ref out

