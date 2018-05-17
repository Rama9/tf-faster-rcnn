#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

case ${DATASET} in
  flickr30k)
    CODE="ground_r_main_msrc_base.py"
    IMFEATDIR="flickr30k_QRCProp_Feat"
    SENDIR="flickr30k_QRCimg_sen_feat"
    TRAIN_IMDB="flickr30k_train_val.lst"
    TEST_IMDB="flickr30k_test.lst"
    LOG_FILE="log/ground_r_supervised_ref.log"
    SAVE_PATH="screenshot/ground_r_supervised_ref_flickr"
    ;;
  referit)
    CODE="ground_r_main_referit_base.py"
    IMFEATDIR="referit_QRCProp_Feat"
    SENDIR="referit_QRCsen_feat"
    TRAIN_IMDB="referit_train_val.lst"
    TEST_IMDB="referit_test.lst"
    LOG_FILE="log/ground_r_supervised_ref.log"
    SAVE_PATH="screenshot/ground_r_supervised_ref_referit"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

set -x

time python ./ground/$CODE \
            --gpu $GPU_ID \
            --net $NET \
            --imFeatDir data/$IMFEATDIR \
            --senDir data/$SENDIR \
            --train_file data/$TRAIN_IMDB \
            --test_file data/$TEST_IMDB \
            --logFile ground/$LOG_FILE \
            --savePath data/$SAVE_PATH
