#!/bin/bash

# number of jobs is slightly greater than 150... it is regions/step + regions%step.
regions=324
njobs=5
step=$(($regions / $njobs))

echo generating X-ray maps using the following parameters

echo regions : $regions
echo njobs : $njobs
echo step : $step

for i in `seq 1 $step $regions`;
    do 
    
    if (( $(( i + $step ))<=$regions)); then
        echo $i $(( i + $step )); 
    fi
    done
echo $i $(( 1 + $regions )); 




