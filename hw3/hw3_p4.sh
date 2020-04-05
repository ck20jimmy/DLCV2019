#!/bin/bash

mkdir model
wget https://www.dropbox.com/s/imzdnx0olgo1bb4/GTA_M2S?dl=1 -O ./model/GTA_M2S
wget https://www.dropbox.com/s/6w4lq3p5rld65af/GTA_S2M?dl=1 -O ./model/GTA_S2M


if [ "$2" = "mnistm" ]
then
	python3 ./ImprovedModel_S2M/predict.py $1 $3
fi


if [ "$2" = "svhn" ]
then
	python3 ./ImprovedModel_M2S/predict.py $1 $3
fi

