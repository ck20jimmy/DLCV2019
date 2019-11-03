#!/bin/bash

# download model

wget https://www.dropbox.com/s/umlx4vmotogr2mo/improved_model?dl=1 -O ./improved_model

# Predict
python3 StrongPredict.py $1 $2 ./improved_model
