#!/bin/bash

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
cd $script_dir

if [ "$#" -ne 1 ]; then
    echo "Usage: test.sh latest"
fi

DSTDIR=$1

mkdir -p $DSTDIR
rm -rf $DSTDIR/*
cp -r ../nnmnkwii/docs/_build/html/* ./$DSTDIR/

find $DSTDIR -type f -name "jp-*" -delete
git status
