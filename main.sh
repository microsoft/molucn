#!/bin/bash

LIST_TARGETS=`cat data/list_targets_20.txt`;

echo $LIST_TARGETS;
echo "explainer: $1";

for target in $LIST_TARGETS; do
    echo "Training $target"
    python xaicode/main.py --target $target --explainer $1
done
