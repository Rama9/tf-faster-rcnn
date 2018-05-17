import tensorflow as tf
import os, sys, argparse
import numpy as np
import time

from dataFeed.dataprovider_flickr30k_msrc_base import dataprovider
from model.ground_r_model_msrc_base import ground_r_model
from util.iou import calc_iou
from util.iou import calc_iou_by_reg_feat

CLASSES = ('__background__', # always index 0
          'people', 'clothing', 'bodyparts', 'animals',
          'vehicles', 'instruments', 'scene', 'other')

class Config(object):
    batch_size = 40
    vocab_size = 17869    
    num_epoch = 3
    max_step = 20000
    optim='adam'
    dropout = 0.5
    lr = 0.001
    weight_decay=0.0
    lstm_dim = 500
    hidden_size = 500
    phrase_len = 19
    num_prop = 100

def update_feed_dict(dataprovider, model, is_train):
    img_feat, sen_feat, gt_reg, bbx_label, pos_all, pos_reg_all = dataprovider.get_next_batch_reg()
    feed_dict = {
                model.sen_data: sen_feat,
                model.vis_data: img_feat,
                model.bbx_label: bbx_label,
                model.gt_reg: gt_reg,
                model.is_train: is_train}
    if dataprovider.multi_reg:
        feed_dict[model.pos_all] = pos_all
        feed_dict[model.pos_reg_all] = pos_reg_all
        feed_dict[model.num_reg] = float(pos_all.shape[0])
    return feed_dict

def eval_cur_batch(gt_label, cur_logits, phr_types, 
    is_train=True, num_sample=0, pos_or_reg=None,
    bbx_loc=None, gt_loc_all=None, ht=1.0, wt=1.0, imgID=None, resDir=None):
    accu = 0.0
    class_accu = np.zeros(len(CLASSES[1:]), dtype=int)
    class_total = np.zeros(len(CLASSES[1:]), dtype=int)
    retRate = 1

    if is_train:
        res_prob = cur_logits[:, :, 0]
        res_label = np.argmax(res_prob, axis=1)        
        accu = float(np.sum(res_label == gt_label)) / float(len(gt_label))
    else:
        num_bbx = len(bbx_loc)
        res_prob = cur_logits[:, :num_bbx, 0]
        #res_label = np.argmax(res_prob, axis=1)        
        out_bbx = np.empty(len(pos_or_reg), dtype=np.object)
        for gt_id in range(len(pos_or_reg)):
            bbxWt = np.zeros((retRate, 4))

            res_labels = np.argsort(res_prob[gt_id])[::-1][:retRate]
            classInd = np.where(np.array(CLASSES[1:]) == phr_types[gt_id])[0]
            cur_gt_pos = gt_label[gt_id]
            success = False
            for res_label in res_labels:
                if pos_or_reg[gt_id] and (res_label in cur_gt_pos):
                    success = True
                    break
            cur_gt = gt_loc_all[gt_id]
            if np.any(cur_gt):
              for resInd, res_label in enumerate(res_labels):
                cur_bbx = bbx_loc[res_label]
                cur_reg = cur_logits[gt_id, res_label, 1:]
                cur_reg[:2] = cur_reg[:2]/2.0
                cur_reg[2:] = -cur_reg[2:]
                iou, oBbx = calc_iou_by_reg_feat(cur_gt, cur_bbx, cur_reg, ht, wt)
                bbxWt[resInd] = oBbx
                if iou > 0.5:
                    success = True
            if success:
                accu += 1.0
                class_accu[classInd] += 1
            class_total[classInd] += 1
            out_bbx[gt_id] = bbxWt
        np.save('%s/%s.npy' %(resDir, imgID), out_bbx)
        accu /= float(num_sample)
    return accu, class_accu, class_total

def load_img_id_list(file_list):
    img_list = []
    with open(file_list) as fin:
        for img_id in fin.readlines():
            img_list.append(int(img_id.strip()))
    img_list = np.array(img_list).astype('int')    
    return img_list

def rec_att_score(sess, dataprovider, model, eval_op, feed_dict, config):
    for img_ind, img_id in enumerate(dataprovider.test_list):
        print '%d/%d: %d'%(img_ind, len(dataprovider.test_list), img_id)
        img_feat_raw, sen_feat_batch, bbx_gt_batch, gt_loc_all, \
        bbx_loc, num_sample_all, pos_or_reg, ht, wt = dataprovider.get_test_feat_reg(img_id)

        if num_sample_all > 0:
            num_sample = len(bbx_gt_batch)
            img_feat = feed_dict[model.vis_data]
            for i in range(num_sample):
                img_feat[i] = img_feat_raw
            sen_feat = feed_dict[model.sen_data]
            sen_feat[:num_sample] = sen_feat_batch

            eval_feed_dict = {
                model.sen_data: sen_feat,
                model.vis_data: img_feat,
                model.is_train: False}

            cur_att_logits = sess.run(eval_op, feed_dict=eval_feed_dict)
            cur_att_logits = cur_att_logits[:num_sample]
            np.save('%s/%d.npy'%(config.rec_dir, img_id), cur_att_logits)

def run_eval(sess, dataprovider, model, eval_op, resDir, feed_dict):
    accu = 0.0
    num_test = 0.0
    num_corr_all = 0.0
    num_cnt_all = 0.0

    if not os.path.isdir(resDir):
        os.makedirs(resDir)

    phrType_accu = np.zeros(len(CLASSES[1:]), dtype=int)
    phrType_total = np.zeros(len(CLASSES[1:]), dtype=int)
    for img_ind, img_id in enumerate(dataprovider.test_list):
        img_feat_raw, sen_feat_batch, bbx_gt_batch, gt_loc_all, \
        bbx_loc, num_sample_all, pos_or_reg, ht, wt, phr_types = dataprovider.get_test_feat_reg(img_id)

        if num_sample_all > 0:
            num_test += 1.0
            num_corr = 0
            num_sample = len(bbx_gt_batch)
            img_feat = feed_dict[model.vis_data]
            for i in range(num_sample):
                img_feat[i] = img_feat_raw
            sen_feat = feed_dict[model.sen_data]
            sen_feat[:num_sample] = sen_feat_batch

            eval_feed_dict = {
                model.sen_data: sen_feat,
                model.vis_data: img_feat,
                model.is_train: False}

            cur_att_logits = sess.run(eval_op, feed_dict=eval_feed_dict)
            cur_att_logits = cur_att_logits[:num_sample]
            cur_accuracy, cur_accu, cur_total = eval_cur_batch(bbx_gt_batch, cur_att_logits, phr_types, False, \
                                                    num_sample_all, pos_or_reg, bbx_loc, gt_loc_all, ht, wt, img_id, resDir)

            num_valid = np.sum(np.all(gt_loc_all, 1))
            print '%d/%d: %d/%d, %.4f'%(img_ind, len(dataprovider.test_list), num_valid, num_sample, cur_accuracy)
            accu += cur_accuracy
            num_corr_all += cur_accuracy*num_sample_all
            num_cnt_all += float(num_sample_all)

            phrType_total += cur_total
            phrType_accu += cur_accu
    accu /= num_test
    accu2 = num_corr_all/num_cnt_all
    print 'Accuracy = %.4f, %.4f'%(accu, accu2)

    for classInd, className in enumerate(CLASSES[1:]):
        print '%s:%.4f ' %(className, float(phrType_accu[classInd])/phrType_total[classInd]),
    print '\n'
    return accu, accu2

def run_training(args):
    train_list = []
    test_list = []
    config = Config()
    train_list = load_img_id_list(args.train_file_list)
    test_list = load_img_id_list(args.test_file_list)

    if args.net == 'res101':
        config.img_feat_size = 2048+5
    elif args.net == 'vgg16':
        config.img_feat_size = 4096+5
    else:
        print 'Not supported netowrk option'
        return
    if not os.path.isdir(args.save_path):
        print 'Model Path does not exist'
        return
    save_path = args.save_path + '/model_%s.ckpt' %args.iterID

    cur_dataset = dataprovider(train_list, test_list, args.img_feat_dir, args.sen_dir, config.vocab_size,
                                batch_size=config.batch_size, phrase_len=config.phrase_len)

    model = ground_r_model(config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

    with tf.Graph().as_default():
        loss, train_op, loss_vec, logits = model.build_model()
        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess,save_path)

        '''
        test code for trainable variables:
        '''
        all_vars = tf.trainable_variables()

        '''
        test end
        '''

        print "-----------------------------------------------"
        feed_dict = update_feed_dict(cur_dataset, model, True)
        #rec_att_score(sess, cur_dataset, model, logits, feed_dict, config)
        eval_accu, eval_accu2 = run_eval(sess, cur_dataset, model, logits, args.resDir, feed_dict)
        print 'ACCURACY IS : ', eval_accu, eval_accu2
        print "-----------------------------------------------"
        model.batch_size = config.batch_size
        cur_dataset.is_save = False

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Grounding Argument PArsing')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default='0')
    parser.add_argument('--net', dest='net', help='Network res101 or vgg16',
                        default='res101')
    parser.add_argument('--imFeatDir', dest='img_feat_dir', help='Visual features of the proposals',
                        default='./')
    parser.add_argument('--senDir', dest='sen_dir', help='Sentence and image statistic features',
                        default='./')
    parser.add_argument('--resDir', dest='resDir', help='Results save directory',
                        default='./')
    parser.add_argument('--train_file', dest='train_file_list', help='File containing training image IDs',
                        default=None)
    parser.add_argument('--test_file', dest='test_file_list', help='File containing test image IDs',
                        default=None)
    parser.add_argument('--iterID', dest='iterID', help='Model iter Id used for testing',
                        default='39')
    parser.add_argument('--savePath', dest='save_path', help='Screeshot directory to save model',
                        default=' ')

    args = parser.parse_args()

    return args

def main(_):
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    run_training(args)

if __name__ == '__main__':
    tf.app.run()
