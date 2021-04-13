#!/bin/bash

# number of jobs is slightly greater than 150... it is regions/step + regions%step
regions=324
njobs=5
step=$(($regions / $njobs))
echo generating X-ray maps using the following parameters

echo regions : $regions
echo njobs : $njobs
echo step : $step

for i in `seq 1 $step $regions`;
    do 
    
    python3 generate-mass-maps.py $i $(( i + $step )) > log$i-update 2> log$i-update &
    
    
    done

python3 generate-mass-maps.py 321 325 > log321-update 2> log321-update &
