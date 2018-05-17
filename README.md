CODE DETAILS:
    This code contains Tensorflow implementation for PGN and QRN modules of QRC Net ( described here). Additional
    support of ResNet architecture is added along with the VGG architecture presented in the paper.

TRAINING:
  PGN:
    (a) We employ the excellently maintained faster-RCNN code for tensorflow by Xinlein Chen. The code can be
    found here with the instructions to download pre-trained models. Alternatively, to download the pretrained models;
    obtain the models from the following link and use the following command to extract it into data folder:
            tar xvzf faster-rcnn_models.tar.gz
    (b) In the next step, the pre-computed sentence features and image-list for flickr30k and referit can be downloaded
    from here and here. Extract the features using the following command into the data folder:
        tar xvzf [flickr30k/referit]-data.tar.gz
    (c) After downloading the above, we finetune the proposal generaton for flickr30k/referit datasets by
    running the following script:
            ./experiments/scripts/finetune_flickr30k.sh $GPU_ID $DATASET[flickr30k/referit] $NET[res101/vgg16]
        To skip this step, models compatible with tensorflow 1.2 could be downloaded from here.
    (d) Finally, we extract the PGN proposals and corresponding visual features by running the following code for both
    train and test images:
            ./experiments/scripts/test_flickr30k.sh $GPU_ID $DATASET[flickr30k/referit] $NET[res101/vgg16]
        Alternately, these steps could be skipped by using the pre-trained sentence features and visual features(*_QRC*_feat);
        obtained from the data archives.


  QRN:
    (a) ground folder contains the code for QRN. To train the grounding model, run the following command.
            ./experiments/scripts/ground_flickr30k.sh $GPU_ID $DATASET[flickr30k/referit] $NET[res101/vgg16]
        This trains a grounding model that learns to choose a proposal that is closest to the query phrase
    (b) To get the results from the trained model; run the following command:
            ./experiments/scripts/ground_test_flickr30k.sh $GPU_ID $DATASET[flickr30k/referit] $NET[res101/vgg16]
        This stores the results in data/<dataset>_ground_result for all the queries of a test image.

TEST:
    (a) To test the image on a random image that does not belong to either flickr30k or referit, first download the
    dictionaries for flickr30k and referit here. Next use the following command with $DATASET pointing to the dataset
    the model is pretrained on:
            ./experiments/scripts/genFeat_flickr30k.sh  $DATASET[flickr30k/referit]
    Modify the $IMINFO path to point to the text file containing test image ids and $IMAGE_DIR to contain the test image
    directory. This step generates the sentence features for test images. 
        For example, a sample image dummy.jpg with queries "dummy test" and "dummy train" should be in the following
        format in the text file:
            dummy dummy_test,dummy_train
    (b) Repeat steps PGN (d) and QRN (b) by modifying the input files accordingly to obtain the output proposals.
