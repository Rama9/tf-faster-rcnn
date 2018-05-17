<p>CODE DETAILS: <br />
    This code contains Tensorflow implementation for PGN and QRN modules of QRC Net ( download <a href="https://arxiv.org/pdf/1708.01676">here</a>). Additional
    support of ResNet architecture is added along with the VGG architecture presented in the paper.  </p>

<p>TRAINING: <br />
     PGN: <br />
        (a) We employ the excellently maintained faster-RCNN code for tensorflow by <a href="https://github.com/endernewton/tf-faster-rcnn">Xinlein Chen</a>. 
            The code can be found here with the instructions to download pre-trained models. 
            Alternatively, to download the pretrained models; obtain the models from the following <a href="https://drive.google.com/file/d/1hDZF-6e5LXEEuhcBrIJhZ4AcKtaAjfRB/view?usp=sharing">link</a> and use the following command to extract it into data folder: <br />
                tar xvzf faster-rcnn<em>models.tar.gz <br />
        (b) In the next step, the pre-computed sentence features and image-list for flickr30k and referit can be downloaded from <a href="https://drive.google.com/file/d/1UcI60Xf4LUTRWA7OqB_FJNPxg6sk661x/view?usp=sharing">here</a> and <a href="https://drive.google.com/file/d/1seluBU4NkUa3R4-ScxqVt9Lcji94rJil/view?usp=sharing">here</a>. 
            Extract the features using the following command into the data folder: <br />
                tar xvzf [flickr30k/referit]-data.tar.gz <br />
        (c) After downloading the above, we finetune the proposal generaton for flickr30k/referit datasets by running the following script: <br />
                ./experiments/scripts/finetune</em>flickr30k.sh $GPU<em>ID $DATASET[flickr30k/referit] $NET[res101/vgg16] <br />
        To skip this step, models compatible with tensorflow 1.2 could be downloaded from here. <br />
        (d) Finally, we extract the PGN proposals and corresponding visual features by running the following code for both train and test images: <br />
                ./experiments/scripts/test</em>flickr30k.sh $GPU<em>ID $DATASET[flickr30k/referit] $NET[res101/vgg16]
        Alternately, these steps could be skipped by using the pre-trained sentence features and visual features(*</em>QRC*_feat); obtained from the data archives.  </p>

<p>QRN:
    (a) ground folder contains the code for QRN. To train the grounding model, run the following command. <br />
            ./experiments/scripts/ground<em>flickr30k.sh $GPU</em>ID $DATASET[flickr30k/referit] $NET[res101/vgg16] <br />
        This trains a grounding model that learns to choose a proposal that is closest to the query phrase <br />
    (b) To get the results from the trained model; run the following command: <br />
            ./experiments/scripts/ground<em>test</em>flickr30k.sh $GPU<em>ID $DATASET[flickr30k/referit] $NET[res101/vgg16] <br />
        This stores the results in data/<dataset></em>ground_result for all the queries of a test image.  </p>

<p>TEST:
    (a) To test the image on a random image that does not belong to either flickr30k or referit, first download the
    dictionaries for flickr30k and referit <a href="https://drive.google.com/file/d/1dt5m1-7mY3FAB6U5xwh1MpzsAZShBHyt/view?usp=sharing">here</a>. Next use the following command with $DATASET pointing to the dataset
    the model is pretrained on: <br />
            ./experiments/scripts/genFeat<em>flickr30k.sh  $DATASET[flickr30k/referit] <br />
    Modify the $IMINFO path to point to the text file containing test image ids and $IMAGE</em>DIR to contain the test image
    directory. This step generates the sentence features for test images. <br />
        For example, a sample image dummy.jpg with queries "dummy test" and "dummy train" should be in the following
        format in the text file: <br />
            dummy dummy<em>test,dummy</em>train <br />
    (b) Repeat steps PGN (d) and QRN (b) by modifying the input files accordingly to obtain the output proposals.  </p>

