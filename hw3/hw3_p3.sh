#!/bin/bash

mkdir model
wget https://www.dropbox.com/s/fi77k8ej70y72i8/DaNN_Mnistm2Svhn?dl=1 -O ./model/DANN_M2S
wget https://www.dropbox.com/s/anaax3za0ec5vwn/DANN_S2M_?dl=1 -O ./model/DANN_S2M


if [ "$2" = "mnistm" ]
then
	python3 ./DANN_svhn2mnistm/predict.py $1 $3
fi


if [ "$2" = "svhn" ]
then
	python3 ./DANN_mnistm2svhn/predict.py $1 $3
fi

