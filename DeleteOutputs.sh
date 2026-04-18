#!/bin/bash

# Files to keep (exact names)
SKIP=("requirements.txt")

for file in *.txt *.png *.csv; do
    [ -e "$file" ] || continue  # skip if no match

    skip=false
    for s in "${SKIP[@]}"; do
        if [[ "$file" == "$s" ]]; then
            skip=true
            break
        fi
    done

    if [ "$skip" = false ]; then
        echo "Deleting: $file"
        rm "$file"
    fi
done