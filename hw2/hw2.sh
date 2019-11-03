#!/bin/bash

# download model
wget https://www.dropbox.com/s/5otzgsiod2wj5cf/baseline_model?dl=1 -O ./baseline_model

# Predict
python3 BaselinePredict.py $1 $2 ./baseline_model
