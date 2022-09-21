#!/bin/bash

LIST_TARGETS=`cat list_targets_350.txt`;

echo $LIST_TARGETS;
echo "pool: $1";
echo "loss: $2";
echo "lambda1: $3";
echo "explainer: $4";

for target in $LIST_TARGETS; do
    echo "Training $target"
    python xaikenza/main.py --target $target --pool $1 --loss $2 --lambda1 $3 --explainer $4
done
