#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
CODE="test_net.py" #DEFAULT CODE

case ${DATASET} in
  flickr30k)
    CODE="extractProps_flickr30k.py"
    TRAIN_IMDB="flickr30k_train_val"
    TEST_IMDB="flickr30k_test"
    IMAGE_DIR="flickr30k-images"
    dataFeatDir="flickr30k_img_sen_feat"
    outFeatDir="flickr30k_QRCProp_Feat"
    outDataFeatDir="flickr30k_QRCimg_sen_feat"
    ITERS=140000
    ;;
  referit)
    CODE="extractProps_referit.py"
    TRAIN_IMDB="referit_train_val"
    TEST_IMDB="referit_test"
    IMAGE_DIR="referit-images"
    dataFeatDir="referit_sen_feat"
    outFeatDir="referit_QRCProp_Feat"
    outDataFeatDir="referit_QRCsen_feat"
    ITERS=50000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

modelPath="output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt"
set -x

time python ./tools/$CODE \
            --gpu $GPU_ID \
            --inDir data/$IMAGE_DIR \
            --featDir data/$dataFeatDir \
            --outFeatDir data/$outFeatDir \
            --outDir data/$outDataFeatDir \
            --model $modelPath \
            --net $NET \
            --testFile data/$TRAIN_IMDB.lst
