#!/bin/bash

functions="bilateral blur sharpen sobel"

if [ "$1" == clean ]
then
  for name in $functions
  do
    rm -rf "$name"{,_cpu{.s,.h},_gpu{.s,.h}}
  done
  exit 0
fi

OUTDIR=${1:-.}
mkdir -p $OUTDIR

for name in $functions
do
  if [ $name.cpp -ot $OUTDIR/$name\_cpu.s ] && \
     [ $name.cpp -ot $OUTDIR/$name\_gpu.s ]
  then
    echo "Skipping generation of halide $name"
    continue
  fi
  echo "Generating halide $name functions"

  g++ -lHalide -o $name $name.cpp
  if [ $? -ne 0 ]
  then
    exit 1
  fi

  for schedule in cpu gpu
  do
    ./$name $schedule halide_$name\_$schedule $OUTDIR/$name\_$schedule
    if [ $? -ne 0 ]
    then
      exit 1
    fi
  done

done
