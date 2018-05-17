import tensorflow as tf
import os, sys, argparse
import numpy as np
import time

from dataFeed.dataprovider_referit_base import dataprovider
from model.ground_r_model_msrc_base import ground_r_model
from util.iou import calc_iou
from util.iou import calc_iou_by_reg_feat

class Config(object):
    batch_size = 40
    img_feat_size = 4096+8
    vocab_size = 8800    
    num_epoch = 3
    max_step = 8000
    optim='adam'
    dropout = 0.5
    lr = 0.001
    weight_decay=0.0
    lstm_dim = 500
    hidden_size = 500
    num_prop = 100

def update_feed_dict(dataprovider, model, is_train):
    img_feat, sen_feat, gt_reg, bbx_label, pos_all, pos_reg_all = dataprovider.get_next_batch_reg()
    # img_feat, sen_feat, bbx_label = dataprovider.get_next_batch()
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

def eval_cur_batch(gt_label, cur_logits, 
    is_train=True, num_sample=0, pos_or_reg=None,
    bbx_loc=None, gt_loc_all=None, ht=1, wt=1):
    accu = 0.0
    retRate = 1

    if is_train:
        res_prob = cur_logits[:, :, 0]
        res_label = np.argmax(res_prob, axis=1)        
        accu = float(np.sum(res_label == gt_label)) / float(len(gt_label))
    else:
        num_bbx = len(bbx_loc)
        res_prob = cur_logits[:, :num_bbx, 0]
        #res_label = np.argmax(res_prob, axis=1)        
        out_bbx = np.empty(len(gt_label), dtype=np.object)
        for gt_id in range(len(gt_label)):
            bbxWt = np.zeros((retRate, 4))

            res_labels = np.argsort(res_prob[gt_id])[::-1][:retRate]
            cur_gt_pos = gt_label[gt_id]
            success = False
            for res_ind, res_label in enumerate(res_labels):
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

            out_bbx[gt_id] = bbxWt
        if num_sample > 0:
            accu /= float(num_sample)
        else:
            accu = 0.0
    return accu, out_bbx

def load_img_id_list(file_list):
    img_list = []
    with open(file_list) as fin:
        for img_id in fin.readlines():
            img_list.append(int(img_id.strip()))
    img_list = np.array(img_list).astype('int')    
    return img_list

def run_eval(sess, dataprovider, model, eval_op, resDir, feed_dict, num_eval=None):
    accu = 0.0
    sum_cor = 0.0
    sum_num = 0.0
    if num_eval is None:
        test_list = dataprovider.test_list
    else:
        test_list = dataprovider.test_list[:num_eval]

    if not os.path.isdir(resDir):
        os.makedirs(resDir)

    for img_ind, img_id in enumerate(test_list):
        img_feat_raw, sen_feat_batch, bbx_gt_batch, gt_loc_all, \
        bbx_loc, num_sample_all, pos_or_reg, ht, wt = dataprovider.get_test_feat_reg(img_id)

        if num_sample_all > 0:
            num_batch = int(sen_feat_batch.shape[0]/model.batch_size)+1
            img_feat = feed_dict[model.vis_data]
            num_corr = 0.0
            out_bbx = np.empty(len(gt_loc_all), dtype=np.object)
            for batch_id in range(num_batch):
                num_sample = min(sen_feat_batch.shape[0], (batch_id+1)*model.batch_size)-batch_id*model.batch_size
                
                for i in range(num_sample):
                    img_feat[i] = img_feat_raw
                sen_feat = feed_dict[model.sen_data]
                sen_feat[:num_sample] = sen_feat_batch[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]
                cur_bbx_gt_batch = bbx_gt_batch[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]
                cur_pos_or_reg = pos_or_reg[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]
                cur_gt_loc_all = gt_loc_all[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]
                # bbx_label[:num_sample] = bbx_label_batch

                eval_feed_dict = {
                    model.sen_data: sen_feat,
                    model.vis_data: img_feat,
                    model.is_train: False}

                cur_att_logits = sess.run(eval_op, feed_dict=eval_feed_dict)
                cur_att_logits = cur_att_logits[:num_sample]
                # print cur_att_logits[:5, :5]

                # cur_gt_label = bbx_label_batch
                cur_accuracy, out_bbx[batch_id*model.batch_size:batch_id*model.batch_size+num_sample] = eval_cur_batch(cur_bbx_gt_batch, cur_att_logits, False, 
                    num_sample, cur_pos_or_reg, bbx_loc, cur_gt_loc_all, ht, wt)
                # cur_accuracy = eval_cur_batch(cur_bbx_gt_batch, cur_att_logits, False, num_sample)
                num_corr += cur_accuracy*num_sample

            cur_accuracy_all = float(num_corr) / float(num_sample_all)
            if img_ind%1000 == 0:
                print '%d/%d: %d/%d, %.4f'%(img_ind, len(test_list), num_sample_all, num_sample, cur_accuracy_all)
            accu += cur_accuracy_all
            sum_cor += float(num_corr)
            sum_num += num_sample_all
            np.save('%s/%s.npy' %(resDir, img_id), out_bbx)

        else:
            print 'No gt for %d'%img_id

    accu /= float(len(test_list))
    accu2 = float(sum_cor)/float(sum_num)
    print 'Accuracy = %.4f, %.4f'%(accu, accu2)
    return accu, accu2

def run_training(args):
    train_list = []
    test_list = []
    config = Config()
    train_list = load_img_id_list(args.train_file_list)
    test_list = load_img_id_list(args.test_file_list)

    if args.net == 'res101':
        config.img_feat_size = 2048+8
    elif args.net == 'vgg16':
        config.img_feat_size = 4096+8
    else:
        print 'Not supported netowrk option'
        return

    if not os.path.isdir(args.save_path):
        print 'Model Path does not exist'
        return
    save_path = args.save_path + '/model_%s.ckpt' %args.iterID

    cur_dataset = dataprovider(train_list, test_list, args.img_feat_dir, args.sen_dir, config.vocab_size,
                                batch_size=config.batch_size)

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
