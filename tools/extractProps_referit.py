#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import cPickle as pickle
from util.iou import calc_iou

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__', # always index 0
           'type1', 'type2', 'type3', 'type4', 'type5', 
           'type6', 'type7', 'type8', 'type9', 'type10',
           'type11', 'type12', 'type13', 'type14', 'type15', 
           'type16', 'type17', 'type18', 'type19', 'type20')

#CLASSES = ('__background__','person','bike','motorbike','car','bus')
def search_pos_in_proposal(gt, proposal, thres=0.5):
    iou_rec = 0.0
    id_rec = -1
    for prop_id, prop in enumerate(proposal):
        # bbx are in form [xmin, ymin, xmax, ymax]
        cur_iou = calc_iou(gt, prop)
        if cur_iou > iou_rec:
            iou_rec = cur_iou
            id_rec = prop_id
    if iou_rec > thres:
        return id_rec
    else:
        return -1

def search_pos_all_in_proposal(gt, proposal, thres=0.5):
    id_rec = []
    for prop_id, prop in enumerate(proposal):
        # bbx are in form [xmin, ymin, xmax, ymax]
        cur_iou = calc_iou(gt, prop)
        if cur_iou > thres:
            id_rec.append(prop_id)
    return id_rec

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name, inDir, featDir, oFeatDir, outDir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(inDir, image_name)

    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    #
    scores, boxes, fc7_feat = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.7
    final_dets = np.zeros((0,5))
    final_feat = np.zeros((0, fc7_feat.shape[1]))
    classLabels = np.zeros(0, dtype=int)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_feat = fc7_feat
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        dets_feat = cls_feat[keep, :]
        final_dets = np.vstack((final_dets, dets))
        final_feat = np.vstack((final_feat, dets_feat))
        classLabels = np.concatenate((classLabels, np.ones(dets.shape[0], dtype=int)*cls_ind))

    numProps = 100
    sInds = np.argsort(final_dets[:,4])[::-1][:numProps]
    props = final_dets[sInds,:4]
    feat = final_feat[sInds,:]
    classLabels = classLabels[sInds]

    print('SAVING : %s of shape : %d %d %d' %(image_name[:-4], props.shape[0], feat.shape[0], feat.shape[1]))
    np.save('%s/%s.npy' %(oFeatDir, image_name[:-4]), feat)

    feat_file = np.load('%s/%s.pkl' %(featDir, image_name[:-4]))
    out_file = '%s/%s.pkl' %(outDir, image_name[:-4])
    feat_keys = ['gt_loc_all', 'ss_box', 'pos_id', 'gt_pos_all', 'bbx_reg_all', 'bbx_reg', 'sen_lang', 'height', 'width', 'sens', 'spat']
    for keyInd, key in enumerate(feat_keys):
        if key == 'ss_box':
            cur_sen_feat = np.load(out_file)
            cur_sen_feat['ss_box'] = props

            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
        elif key == 'pos_id':
            cur_sen_feat = np.load(out_file)
            y = []
            for gt_box in cur_sen_feat['gt_loc_all']:
                if sum(gt_box) == 0:
                    y.append(-1)
                else:
                    cur_pos = search_pos_in_proposal(gt_box, props, 0.5)
                    y.append(cur_pos)
            cur_sen_feat['pos_id'] = y        
            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
        elif key == 'gt_pos_all':
            cur_sen_feat = np.load(out_file)
            gt_pos_all = []
            for gt_box in cur_sen_feat['gt_loc_all']:
                if sum(gt_box) == 0:
                    gt_pos_all.append([])
                else:
                    pos_all = search_pos_all_in_proposal(gt_box, props, 0.5)
                    gt_pos_all.append(pos_all)
            cur_sen_feat['gt_pos_all'] = gt_pos_all
            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
        elif key == 'bbx_reg_all':
            pos_ids_all = cur_sen_feat['gt_pos_all']
            img_reg = []
            ss = props
            for pos_ind, pos_ids in enumerate(pos_ids_all):
                if pos_ids == []:
                    img_reg.append([])
                else:
                    gt_box = cur_sen_feat['gt_loc_all'][pos_ind].astype('float')
                    cur_img_reg_all = np.zeros((len(pos_ids), 4)).astype('float')
                    for cur_ind, pos_id in enumerate(pos_ids):
                        cur_img_reg = np.zeros(4)
                        cur_img_reg[0] = (gt_box[0]+gt_box[2]-ss[pos_id,0]-ss[pos_id,2])/(ss[pos_id,2]-ss[pos_id,0]+1.0)
                        cur_img_reg[1] = (gt_box[1]+gt_box[3]-ss[pos_id,1]-ss[pos_id,3])/(ss[pos_id,3]-ss[pos_id,1]+1.0)
                        cur_img_reg[2] = np.log((ss[pos_id,2]-ss[pos_id,0]+1.0)/(gt_box[2]-gt_box[0]+1.0))
                        cur_img_reg[3] = np.log((ss[pos_id,3]-ss[pos_id,1]+1.0)/(gt_box[3]-gt_box[1]+1.0))
                        cur_img_reg_all[cur_ind] = cur_img_reg
                    img_reg.append(cur_img_reg_all)
            cur_sen_feat['bbx_reg_all'] = img_reg
            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
        elif key == 'bbx_reg':
            cur_sen_feat = np.load(out_file)
            gt_box = cur_sen_feat['gt_loc_all']
            ss = props
            pos_ids = cur_sen_feat['pos_id']
            img_reg = np.zeros((len(pos_ids), 4))
            for pos_ind, pos_id in enumerate(pos_ids):
                if pos_id == -1:
                    img_reg[pos_ind] = [0.0, 0.0, 0.0, 0.0]
                else:
                    gt_box = cur_sen_feat['gt_loc_all'][pos_ind]
                    img_reg[pos_ind, 0] = float((gt_box[0]+gt_box[2])-(ss[pos_id, 0]+ss[pos_id, 2])) / float(ss[pos_id, 2]-ss[pos_id, 0]+1.0)
                    img_reg[pos_ind, 1] = float((gt_box[1]+gt_box[3])-(ss[pos_id, 1]+ss[pos_id, 3])) / float(ss[pos_id, 3]-ss[pos_id, 1]+1.0)
                    img_reg[pos_ind, 2] = np.log((ss[pos_id,2]-ss[pos_id,0]+1.0)/(gt_box[2]-gt_box[0]+1.0))
                    img_reg[pos_ind, 3] = np.log((ss[pos_id,3]-ss[pos_id,1]+1.0)/(gt_box[3]-gt_box[1]+1.0))
            cur_sen_feat['bbx_reg'] = img_reg
            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        elif key == 'spat':
            cur_sen_feat = np.load(out_file)
            bbx = cur_sen_feat['ss_box']
            height = cur_sen_feat['height']
            width = cur_sen_feat['width']
            spat_feat = np.zeros((bbx.shape[0], 8))
            spat_feat[:, :4] = bbx.astype('float')          # unnormalized [xmin, ymin, xmax, ymax]
            spat_feat[:, 0] = spat_feat[:, 0]*2.0/float(width) - 1.0
            spat_feat[:, 1] = spat_feat[:, 1]*2.0/float(height) - 1.0
            spat_feat[:, 2] = spat_feat[:, 2]*2.0/float(width) - 1.0
            spat_feat[:, 3] = spat_feat[:, 3]*2.0/float(height) - 1.0
            spat_feat[:, 4] = 0.5*(spat_feat[:, 0]+spat_feat[:, 2])
            spat_feat[:, 5] = 0.5*(spat_feat[:, 1]+spat_feat[:, 3])
            spat_feat[:, 6] = spat_feat[:, 2]-spat_feat[:,0]
            spat_feat[:, 7] = spat_feat[:, 3]-spat_feat[:, 1]
            cur_sen_feat['spat'] = spat_feat
            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        elif key == 'gt_loc_all':
            cur_sen_feat = {}
            oFeat = feat_file['gt_loc_all']
            cur_sen_feat['gt_loc_all'] = oFeat
            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
        else:
            cur_sen_feat = np.load(out_file)
            cur_sen_feat[key] = feat_file[key]
            pickle.dump(cur_sen_feat, open(out_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL) 

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--inDir', dest='inputDir', help='Directory for demo images',
                        default='./')
    parser.add_argument('--outDir', dest='outputDir', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--featDir', dest='featureDir', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--outFeatDir', dest='outFeatDir', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--threadID', dest='threadId', help='Thread ID for extract',
                        default=0)
    parser.add_argument('--testFile', dest='referittest', help='Referit test IDs',
                        default=None)
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' %args.gpu_id

    # model path
    demonet = args.demo_net
    #dataset = args.dataset
    tfmodel = args.model #os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.50

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    referittestIDs = np.loadtxt(args.referittest, dtype=np.object)

    if not os.path.exists(args.outputDir):
        print('CREATING DIR')
        os.mkdir(args.outputDir)
    if not os.path.exists(args.outFeatDir):
        print('CREATING DIR')
        os.mkdir(args.outFeatDir)

    for imInd, imID in enumerate(referittestIDs):
        im_name = '%s.jpg' %imID
        if imInd%1000 == 0:
            print('%d/%d Images processed..' %(imInd, referittestIDs.shape[0]))

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name, args.inputDir, args.featureDir, args.outFeatDir, args.outputDir)
        
    plt.show()
