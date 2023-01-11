#!/bin/bash

# number of jobs is slightly greater than 150... it is regions/step + regions%step
regions=324
njobs=50
step=$(($regions / $njobs))
echo generating X-ray maps using the following parameters

echo regions : $regions
echo njobs : $njobs
echo step : $step

for i in `seq 1 $step $regions`;
    do 
    
    if (( $(( i + $step ))<=$regions)); then
        echo $i $(( i + $step )); 
        python3 gizmo-generate-sz-maps.py $i $(( i + $step )) > log$i-update 2> log$i-update &
    fi
    done
echo $i $(( 1 + $regions )); 
python3 gizmo-generate-sz-maps.py 319 325 > log322-update 2> log322-update &