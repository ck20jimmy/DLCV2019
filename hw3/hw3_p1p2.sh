#!/bin/bash

mkdir ./model
wget https://www.dropbox.com/s/xot0meh4yya6sc1/ACGAN?dl=1 -O ./model/ACGAN
wget https://www.dropbox.com/s/w4wa12i1bu2bm5x/ColorModel?dl=1 -O ./model/ColorModel
wget https://www.dropbox.com/s/pj9x8b41sjukyze/ACGAN_noise.npy?dl=1 -O ./model/ACGAN_noise.npy

wget https://www.dropbox.com/s/rp310whkkh9rtdb/DCGAN?dl=0 -O ./model/DCGAN
wget https://www.dropbox.com/s/njzsb7u46qq2clu/good_noise_32.npy?dl=0 -O ./model/DCGAN_noise.npy

python3 ./DCGAN/predict.py $1
python3 ./ACGAN/predict.py $1