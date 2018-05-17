#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
echo $DATASET
CODE="trainval_net.py" #DEFAULT CODE

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  flickr30k)
    TRAIN_IMDB="flickr30k_train_val"
    TEST_IMDB="flickr30k_test"
    STEPSIZE="[100000]"
    ITERS=140000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    CODE="finetune_flickrNet.py"
    ;;
  referit)
    TRAIN_IMDB="referit_train_val"
    TEST_IMDB="referit_test"
    STEPSIZE="[35000]"
    ITERS=50000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    CODE="finetune_referit.py"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/customClass/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/$CODE \
            --weight data/voc_2007_trainval+voc_2012_trainval/${NET}_faster_rcnn_iter_110000.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET}.yml \
            --tag ${EXTRA_ARGS_SLUG} \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/$CODE \
            --weight data/voc_2007_trainval+voc_2012_trainval/${NET}_faster_rcnn_iter_110000.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET}.yml \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    fi
fi

#./experiments/scripts/test_faster_rcnn.sh $@
