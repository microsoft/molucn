#!/bin/bash

LIST_TARGETS=`cat list_targets_350.txt`;

echo $LIST_TARGETS;
echo "explainer: $1";

for target in $LIST_TARGETS; do
    echo "Training $target"
    python xaikenza/main.py --target $target --explainer $1
done
