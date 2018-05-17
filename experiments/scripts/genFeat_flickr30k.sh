#!/bin/bash

set -x
set -e

DATASET=$1

case ${DATASET} in
  flickr30k)
    CODE="gen_flickr30k_testFeat.py"
    IMINFO="sample_flickr30k.lst"
    IMAGE_DIR="flickr30k-images"
    RESDIR="SampleFeat"
    DICT="dict/word_dict_flickr30k_global.pkl"
    ;;
  referit)
    CODE="gen_referit_testFeat.py"
    IMINFO="sample_referit.lst"
    IMAGE_DIR="referit-images"
    RESDIR="SampleReferITFeat"
    DICT="dict/word_dict_referit.npy"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

time python ./tools/$CODE \
            --imInfo data/$IMINFO \
            --imgDir data/$IMAGE_DIR \
            --outDir data/$RESDIR \
            --dict data/$DICT
