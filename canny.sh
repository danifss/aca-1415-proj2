#!/bin/bash
rm .ref.swp .out.swp
cuda-memcheck ./canny -i $1.pgm -o $1Out.pgm
hexdump $1Out.pgm > out
hexdump reference.pgm > ref
vim -d ref out
