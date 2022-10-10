#!/bin/bash

LIST_TARGETS=`cat data/list_targets_350.txt`;

echo "Using attribution technique: ${1}";

for target in ${LIST_TARGETS}; do
    echo "Now training and attributing on target w. PDB Id: ${target}"
    python molucn/main.py --target ${target} --explainer ${1}
done
