# TODO: create shell script for Problem 1

#!/bin/bash

wget https://www.dropbox.com/s/s125qm0zkd6yrfg/Task1_model?dl=1 -O ./Task1_model
python3 ./code/Task1/predict.py $1 $2 $3 ./Task1_model
