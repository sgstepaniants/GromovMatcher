#!/bin/bash 

# Run the script
declare -i batches=180

for (( i = 1; i <= $batches; i++ ))
do
    sbatch --cpus-per-task 4 -o M2Sresults-$i-$batches.log run_M2S.sh $i $batches
done
