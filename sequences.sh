#!/bin/bash

# number of jobs
regions=324
njobs=50
step=$(($regions / $njobs))
echo generating X-ray maps using the following parameters

echo regions : $regions
echo njobs : $njobs
echo step : $step

for i in `seq 1 $step $regions`;
    do 
    
    echo $i; 
    
    done
