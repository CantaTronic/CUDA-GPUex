#!/bin/sh

# do not forget to cd into working directory
cd ch1_simple && make targ=$1 && ./$1; cd ..

