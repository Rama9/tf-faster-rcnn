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
    img_feat_size = 2048+8
    vocab_size = 8800    
    num_epoch = 3
    max_step = 8000
    optim='adam'
    dropout = 0.5
    lr = 0.001
    weight_decay=0.0
    lstm_dim = 500
    hidden_size = 500

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
    if is_train:
        res_prob = cur_logits[:, :, 0]
        res_label = np.argmax(res_prob, axis=1)        
        accu = float(np.sum(res_label == gt_label)) / float(len(gt_label))
    else:
        num_bbx = len(bbx_loc)
        res_prob = cur_logits[:, :num_bbx, 0]
        res_label = np.argmax(res_prob, axis=1)        
        for gt_id in range(len(gt_label)):
            cur_gt_pos = gt_label[gt_id]
            success = False
            if pos_or_reg[gt_id] and (res_label[gt_id] in cur_gt_pos):
                success = True        
            cur_gt = gt_loc_all[gt_id]
            if np.all(cur_gt):
                cur_bbx = bbx_loc[res_label[gt_id]]
                cur_reg = cur_logits[gt_id, res_label[gt_id], 1:]
                iou, _ = calc_iou_by_reg_feat(cur_gt, cur_bbx, cur_reg, ht, wt)
                if iou > 0.5:
                    success = True
            if success:
                accu += 1.0
        if num_sample > 0:
            accu /= float(num_sample)
        else:
            accu = 0.0
    return accu

def load_img_id_list(file_list):
    img_list = []
    with open(file_list) as fin:
        for img_id in fin.readlines():
            img_list.append(int(img_id.strip()))
    img_list = np.array(img_list).astype('int')    
    return img_list

def run_eval(sess, dataprovider, model, eval_op, feed_dict, num_eval=None):
    accu = 0.0
    sum_cor = 0.0
    sum_num = 0.0
    if num_eval is None:
        test_list = dataprovider.test_list
    else:
        test_list = dataprovider.test_list[:num_eval]

    for img_ind, img_id in enumerate(test_list):
        img_feat_raw, sen_feat_batch, bbx_gt_batch, gt_loc_all, \
        bbx_loc, num_sample_all, pos_or_reg, ht, wt = dataprovider.get_test_feat_reg(img_id)

        if num_sample_all > 0:
            num_batch = int(sen_feat_batch.shape[0]/model.batch_size)+1
            img_feat = feed_dict[model.vis_data]
            num_corr = 0.0
            for batch_id in range(num_batch):
                num_sample = min(sen_feat_batch.shape[0], (batch_id+1)*model.batch_size)-batch_id*model.batch_size
                
                for i in range(num_sample):
                    img_feat[i] = img_feat_raw
                sen_feat = feed_dict[model.sen_data]
                sen_feat[:num_sample] = sen_feat_batch[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]
                cur_bbx_gt_batch = bbx_gt_batch[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]
                cur_pos_or_reg = pos_or_reg[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]
                cur_gt_loc_all = gt_loc_all[batch_id*model.batch_size:batch_id*model.batch_size+num_sample]

                eval_feed_dict = {
                    model.sen_data: sen_feat,
                    model.vis_data: img_feat,
                    model.is_train: False}

                cur_att_logits = sess.run(eval_op, feed_dict=eval_feed_dict)
                cur_att_logits = cur_att_logits[:num_sample]

                cur_accuracy = eval_cur_batch(cur_bbx_gt_batch, cur_att_logits, False, 
                    num_sample, cur_pos_or_reg, bbx_loc, cur_gt_loc_all, ht, wt)
                num_corr += cur_accuracy*num_sample

            cur_accuracy_all = float(num_corr) / float(num_sample_all)
            print '%d/%d: %d/%d, %.4f'%(img_ind, len(test_list), num_sample_all, num_sample, cur_accuracy_all)
            accu += cur_accuracy_all
            sum_cor += float(num_corr)
            sum_num += num_sample_all

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
        print 'Save models into %s'%args.save_path
        os.makedirs(args.save_path)
    logDir = args.log_file[:-len(args.log_file.split('/')[-1:][0])]
    if not os.path.isdir(logDir):
        print logDir
        os.makedirs(logDir)
    log = open(args.log_file, 'w', 0)

    cur_dataset = dataprovider(train_list, test_list, args.img_feat_dir, args.sen_dir, config.vocab_size,
                                img_feat_size=config.img_feat_size, batch_size=config.batch_size)

    model = ground_r_model(config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    with tf.Graph().as_default():
        loss, train_op, loss_vec, logits = model.build_model()
        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        duration = 0.0

        for step in xrange(config.max_step):
            start_time = time.time()
            feed_dict = update_feed_dict(cur_dataset, model, True)
            _,loss_value,loss_vec_value, cur_logits = sess.run([train_op, loss, loss_vec, logits], feed_dict=feed_dict)
            duration += time.time()-start_time

            if cur_dataset.is_save:
                print 'Save model_%d into %s'%(cur_dataset.epoch_id, args.save_path)
                saver.save(sess, '%s/model_%d.ckpt'%(args.save_path, cur_dataset.epoch_id))
                cur_dataset.is_save = False
 
                if cur_dataset.epoch_id%5 == 0:
                    eval_accu, eval_accu2 = run_eval(sess, cur_dataset, model, logits, feed_dict)                
                    log.write('%d/%d: %.4f, %.4f, %.4f\n'%(step+1, cur_dataset.epoch_id, loss_value, eval_accu, eval_accu2))
                    print "-----------------------------------------------"
                if cur_dataset.epoch_id == 40:
                    break
                
    log.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Grounding Argument Parsing')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default='0')
    parser.add_argument('--net', dest='net', help='Network res101 or vgg16',
                        default='res101')
    parser.add_argument('--imFeatDir', dest='img_feat_dir', help='Visual features of the proposals',
                        default='./')
    parser.add_argument('--senDir', dest='sen_dir', help='Sentence and image statistic features',
                        default='./')
    parser.add_argument('--train_file', dest='train_file_list', help='File containing training image IDs',
                        default=None)
    parser.add_argument('--test_file', dest='test_file_list', help='File containing test image IDs',
                        default=None)
    parser.add_argument('--logFile', dest='log_file', help='Path for the training output log',
                        default='temp.log')
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
