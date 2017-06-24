#!/bin/sh
set -x # display shell commands prior to execution


path=~/git/KT8302_notes/tutorials

# copy movies that follow mov-ch?-convention
for nbook in $( ls -d *.ipynb ); do
    cp $path/$nbook .
    cp $path/figs/* ./figs/
done





