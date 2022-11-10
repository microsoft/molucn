#!/bin/bash

LIST_TARGETS=`cat data/list_targets_350.txt`;

echo ${LIST_TARGETS};

for target in ${LIST_TARGETS}; do
    echo "Now training and attributing on target ${target}"
    python molucn/main_rf.py --target ${target}
done
