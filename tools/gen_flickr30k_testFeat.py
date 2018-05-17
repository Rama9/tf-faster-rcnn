import numpy as np
import os, sys, re, argparse
import cPickle as pickle
import scipy.io
import skimage.io
from util.iou import calc_iou
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_sen_file(sen_file):
    sens = []
    p_ids = []
    sen_ids = []
    p_types = []
    with open(sen_file) as fin:
        for cur_line in fin.readlines():
            cons = cur_line.strip().split(':')
            sen = ':'.join(cons[:-1])
            sens.append(sen.lower())
            p_id = int(cons[-1].split('#')[0])
            sen_id = int(cons[-1].split('#')[1].split('$')[0])
            p_type = cons[-1].split('#')[1].split('$')[1]
            p_ids.append(p_id)
            sen_ids.append(sen_id)
            p_types.append(p_type)
    return sens, p_ids, sen_ids, p_types

def update_cap_dict(sens, word_dict):
    for sen in sens:
        words = sen.split()
        for word in words:
            if word not in word_dict:
                # word id starts from 0, word_dict[word] = [word_id, num_appear]
                word_dict[word] = [len(word_dict), 1]  
            else:
                word_dict[word][1] += 1 # appear times plus one
    return word_dict

def tokenize_img_cap(sens, word_dict):
    tokens = []
    for sen in sens:
        words = sen.split('_')
        token = []
        for word in words:
            token.append(word_dict[word][0])
        tokens.append(token)
    return tokens

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

def gen_img_cap_feat2(img_list, gt_file, proposal_dir, sen_dir, res_dir):
    word_dict = {}
    for img_ind, img_id in enumerate(img_list):
        img_id = img_id.rstrip()
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)
        ss = scipy.io.loadmat('%s/%s.mat'%(proposal_dir, img_id))['cur_bbxes']    # [hmin, wmin, hmax, wmax]
        # original proposal is [hmin, wmin, hmax, wmax] --> [xmin, ymin, xmax, ymax]
        ss = ss[:, [1, 0, 3, 2]] -1 

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        sens = cur_sen_feat['sen_lang']
        gt_boxes = cur_sen_feat['gt_box']
        word_dict = np.load('word_dict_flickr30k_global.pkl')
        tokens = tokenize_img_cap(sens, word_dict)
        print tokens

        y = []
        for gt_box in gt_boxes:
            cur_gt = gt_box
            cur_pos = search_pos_in_proposal(cur_gt, ss, 0.5)
            y.append(cur_pos)

        #cur_sen_feat = {}
        cur_sen_feat['sens'] = tokens
        cur_sen_feat['pos_id'] = y
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    return word_dict

def add_img_gt_label2(img_list, gt_file, proposal_dir, sen_dir, res_dir):
    for img_ind, img_id in enumerate(img_list):
        img_id = img_id.rstrip()
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)
        ss = scipy.io.loadmat('%s/%s.mat'%(proposal_dir, img_id))['cur_bbxes']    # [hmin, wmin, hmax, wmax]
        # original proposal is [hmin, wmin, hmax, wmax] --> [xmin, ymin, xmax, ymax]
        ss = ss[:, [1, 0, 3, 2]] -1 

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))

        gt_pos_all = []
        for gt_box in cur_sen_feat['gt_box']:
            cur_gt = gt_box

        #cur_sen_feat = np.load('%s/protest_img_%s.pkl'%(res_dir, img_id))
            #gt_box.append(cur_gt)
            pos_all = search_pos_all_in_proposal(cur_gt, ss, 0.5)
            gt_pos_all.append(pos_all)

        cur_sen_feat['gt_pos_all'] = gt_pos_all
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def gen_img_cap_feat(img_list, gt_file, proposal_dir, res_dir, word_dict, queries=None):
    for img_ind, img_id in enumerate(img_list):
        #For World Protest Images
        print '%d/%d: %s; genFeat'%(img_ind+1, len(img_list), img_id)
        key = img_id
        try:
            gt_box = np.load(gt_file)[key]        # [xmin, ymin, xmax, ymax]
        except:
            gt_box = [0, 0, 0 , 0]

        if proposal_dir != '':
            ss = scipy.io.loadmat('%s/%s.mat'%(proposal_dir, img_id))['cur_bbxes']    # [hmin, wmin, hmax, wmax]
            # original proposal is [hmin, wmin, hmax, wmax] --> [xmin, ymin, xmax, ymax]
            ss = ss[:, [1, 0, 3, 2]] -1
        else:
            ss = np.zeros((100, 4))

        #sens = np.array(queries[img_ind].split(','))
        #tokens = tokenize_img_cap(sens, word_dict)
        #print tokens

        y = []
        #cur_token = tokens[0]
        cur_gt = gt_box
        cur_pos = search_pos_in_proposal(cur_gt, ss, 0.5)
        y.append(cur_pos)

        cur_sen_feat = {}
        #cur_sen_feat['sens'] = tokens
        cur_sen_feat['pos_id'] = y
        cur_sen_feat['ss_box'] = ss
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    return word_dict

def add_img_gt_label(img_list, gt_file, proposal_dir, res_dir, queries=None):
    for img_ind, img_id in enumerate(img_list):
        print '%d/%d: %s; GT label'%(img_ind+1, len(img_list), img_id)
        key = img_id
        try:
            gt_box = np.load(gt_file)[key]        # [xmin, ymin, xmax, ymax]
            print 'GT Assigned'
        except:
            gt_box = np.array([0, 0, 0, 0])

        sens = np.array(queries[img_ind].split(','))
        cur_gt = gt_box

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        ss = cur_sen_feat['ss_box']
        gt_box = []
        gt_pos_all = []
        gt_box.append(cur_gt)
        pos_all = search_pos_all_in_proposal(cur_gt, ss, 0.5)
        gt_pos_all.append(pos_all)

        gt_box = np.array(gt_box).astype('int')
        cur_sen_feat['gt_box'] = gt_box
        cur_sen_feat['sen_lang'] = sens
        cur_sen_feat['gt_pos_all'] = gt_pos_all
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def add_sen_id_label(img_list, res_dir):
    for img_ind, img_id in enumerate(img_list):
        #img_id = '%s_mdf_%s' %(img_id.split(' ')[0].split('_')[1], img_id.split(' ')[1].rstrip())
        img_id = img_id.rstrip()
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        cur_sen_feat['sen_ids'] = np.arange(cur_sen_feat['sen_lang'].shape[0])
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    

def add_semantic_feat(img_list, gt_dir, context_dir, sen_dir, res_dir):
    word_dict = np.load('word_dict_flickr30k_global.pkl')
    import nltk
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    for img_ind, img_id in enumerate(img_list):
        img_id = img_id.strip().split('/')[-1][:-4]
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        cur_con_feat = np.load('%s/%s.pkl'%(context_dir, img_id))
        con_rec = []
        for p_id in cur_con_feat.keys():
            if len(cur_con_feat[p_id]) > 0:
                for con in cur_con_feat[p_id]:
                    word = con[1]
                    new_word = wordnet_lemmatizer.lemmatize(word, pos='n')        # lemmatize in noun
                    new_word = wordnet_lemmatizer.lemmatize(new_word, pos='v')    # lemmatize in verb                    
                    con_rec.append([p_id, con[0], word_dict[new_word][0], con[2]])
        con_rec = np.array(con_rec).astype('int')
        cur_sen_feat['context_triplet'] = con_rec
        #pickle.dump(cur_sen_feat, open('%s/protest_img_%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def check_semantic_feat(img_list, gt_dir, context_dir, sen_dir, res_dir):
    abnormal_id = []
    num_all = 0
    num_cor = 0
    for img_ind, img_id in enumerate(img_list):
        img_id = img_id.strip().split('/')[-1][:-4]
        # print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        if len(cur_sen_feat['context_triplet']) > 0:
            num_all += len(cur_sen_feat['context_triplet'])
            pos_ids = np.array(cur_sen_feat['pos_id']).astype('int')
            pos_ind = np.where(pos_ids != -1)[0]

            num_neg = 0
            for con in cur_sen_feat['context_triplet']:
                if con[0] not in pos_ind:
                    num_neg += 1
            num_cor += len(cur_sen_feat['context_triplet'])-num_neg
            if num_neg == len(cur_sen_feat['context_triplet']):
                abnormal_id.append(int(img_id))
    # np.save('abnormal_id_con_feat.npy', abnormal_id)
    print '%d/%d'%(num_cor, num_all)

def add_img_hw_label(img_list, img_dir, res_dir):
    for img_ind, img_id in enumerate(img_list):
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        if os.path.exists('%s/%s.jpg'%(img_dir, img_id)):
            I = skimage.io.imread('%s/%s.jpg'%(img_dir, img_id))
        else:
            I = skimage.io.imread('%s/%s.png'%(img_dir, img_id))
        #print I.shape
        if len(I.shape) == 3:
            h, w, c = I.shape
        elif len(I.shape) == 4:
            a, h, w, c = I.shape
        else:
            h,w = I.shape

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        cur_sen_feat['height'] = h
        cur_sen_feat['width'] = w

        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    

def gen_global_dict(img_list, gt_dir, proposal_dir, sen_dir, res_dir):
    import nltk
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    word_dict = {}
    word_id = 0

    for img_ind, img_id in enumerate(img_list):
        img_id = img_id.strip().split('/')[-1][:-4]
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        sen_lang = []
        with open('/home/kanchen/medifor/sen_lang/%s.txt'%img_id) as fin:
            for cur_line in fin.readlines():
                sen_lang.append(cur_line.strip())
        for sen in sen_lang:
            # remove non-ascii chars
            sen = ''.join([i if ord(i) < 128 else ' ' for i in sen])
            words = sen.split()
            for word in words:
                new_word = wordnet_lemmatizer.lemmatize(word, pos='n')        # lemmatize in noun
                new_word = wordnet_lemmatizer.lemmatize(new_word, pos='v')    # lemmatize in verb
                if new_word not in word_dict:
                    word_dict[new_word] = [word_id, 1]
                    word_id += 1
                else:
                    word_dict[new_word][1] += 1
    word_dict['<unk>'] = [word_id, -1]
    print 'vocab size: %d'%len(word_dict)
    num_cnt = np.zeros((20))
    for word in word_dict:
        if word_dict[word][1] < 21 and word_dict[word][1] > 0:
            num_cnt[word_dict[word][1]-1] += 1
    print num_cnt
    pickle.dump(word_dict, open('word_dict_flickr30k_global.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def add_img_sen_lang_token(img_list, gt_dir, word_dict, proposal_dir, res_dir):
    import nltk
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    for img_ind, img_id in enumerate(img_list):
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        sen_tokens = []
        for sen in cur_sen_feat['sen_lang']:
            sen = ''.join([i if ord(i) < 128 else ' ' for i in sen])
            words = sen.split('_')
            sen_token = []
            for word in words:
                new_word = wordnet_lemmatizer.lemmatize(word, pos='n')        # lemmatize in noun
                new_word = wordnet_lemmatizer.lemmatize(new_word, pos='v')    # lemmatize in verb
                if new_word in word_dict:
                    sen_token.append(word_dict[new_word][0])
                else:
                    print 'ENTERS <UNK>!!!!!!!!!!!!'
                    print new_word
                    sen_token.append(word_dict['<unk>'][0])
            sen_tokens.append(sen_token)
        cur_sen_feat['sen_lang_token'] = sen_tokens
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def add_img_reg_label(img_list, res_dir):
    for img_ind, img_id in enumerate(img_list):
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        h = cur_sen_feat['height']
        w = cur_sen_feat['width']
        ss = cur_sen_feat['ss_box']

        pos_ids = cur_sen_feat['pos_id']
        img_reg = np.zeros((len(pos_ids), 4))
        for pos_ind, pos_id in enumerate(pos_ids):
            if pos_id == -1:
                img_reg[pos_ind] = [0.0, 0.0, 0.0, 0.0]
            else:
                gt_box = cur_sen_feat['gt_box'][pos_ind]    # [xmin, ymin, xmax, ymax]
                img_reg[pos_ind, 0] = float((gt_box[0]+gt_box[2])-(ss[pos_id, 0]+ss[pos_id, 2])) / float(ss[pos_id, 2]-ss[pos_id, 0]+1.0)
                img_reg[pos_ind, 1] = float((gt_box[1]+gt_box[3])-(ss[pos_id, 1]+ss[pos_id, 3])) / float(ss[pos_id, 3]-ss[pos_id, 1]+1.0)
                img_reg[pos_ind, 2] = np.log((ss[pos_id, 2]-ss[pos_id, 0]+1.0)/(gt_box[2]-gt_box[0]+1.0))
                img_reg[pos_ind, 3] = np.log((ss[pos_id, 3]-ss[pos_id, 1]+1.0)/(gt_box[3]-gt_box[1]+1.0))

        cur_sen_feat['bbx_reg'] = img_reg

        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)        

def add_img_reg_label_all(img_list, res_dir):
    for img_ind, img_id in enumerate(img_list):
        #img_id = img_id.split(' ')[0]
        #img_id = '%s_mdf_%s' %(img_id.split(' ')[0].split('_')[1], img_id.split(' ')[1].rstrip())
        img_id = img_id.rstrip()
        print '%d/%d: %s'%(img_ind+1, len(img_list), img_id)        

        #cur_sen_feat = np.load('%s/protest_img_%s.pkl'%(res_dir, img_id))
        cur_sen_feat = np.load('%s/%s.pkl'%(res_dir, img_id))
        h = cur_sen_feat['height']
        w = cur_sen_feat['width']
        ss = cur_sen_feat['ss_box']

        pos_ids_all = cur_sen_feat['gt_pos_all']
        img_reg = []
        for pos_ind, pos_ids in enumerate(pos_ids_all):
            if pos_ids == []:
                img_reg.append([])
            else:
                gt_box = cur_sen_feat['gt_box'][pos_ind].astype('float')    # [xmin, ymin, xmax, ymax]
                cur_img_reg_all = np.zeros((len(pos_ids), 4)).astype('float')
                for cur_ind, pos_id in enumerate(pos_ids):
                    cur_img_reg = np.zeros(4)
                    cur_img_reg[0] = (gt_box[0]+gt_box[2]-ss[pos_id, 0]-ss[pos_id, 2]) / (ss[pos_id, 2]-ss[pos_id, 0]+1.0)
                    cur_img_reg[1] = (gt_box[1]+gt_box[3]-ss[pos_id, 1]-ss[pos_id, 3]) / (ss[pos_id, 3]-ss[pos_id, 1]+1.0)
                    cur_img_reg[2] = np.log((ss[pos_id, 2]-ss[pos_id, 0]+1.0)/(gt_box[2]-gt_box[0]+1.0))
                    cur_img_reg[3] = np.log((ss[pos_id, 3]-ss[pos_id, 1]+1.0)/(gt_box[3]-gt_box[1]+1.0))
                    cur_img_reg_all[cur_ind] = cur_img_reg
                img_reg.append(cur_img_reg_all)

        cur_sen_feat['bbx_reg_all'] = img_reg

        #pickle.dump(cur_sen_feat, open('%s/protest_img_%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(cur_sen_feat, open('%s/%s.pkl'%(res_dir, img_id), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Generating sentence features')
    parser.add_argument('--imInfo', dest='img_info', help='List of test Images and respectivequeries',
                        default=None)
    parser.add_argument('--gtInfo', dest='gt_file', help='Annotations ground truth of test images[Optional]',
                        default=None)
    parser.add_argument('--imgDir', dest='img_dir', help='Directory containing images',
                        default=None)
    parser.add_argument('--outDir', dest='res_dir', help='Output directory',
                        default='./')
    parser.add_argument('--propDir', dest='proposal_dir', help='Proposals for the test images[Optional]',
                        default='')
    parser.add_argument('--dict', dest='dictFile', help='Word dictionary mapping for the dataset',
                        default=None)
    args = parser.parse_args()

    return args


def main():
        args = parse_args()
        img_list = np.loadtxt(args.img_info, dtype=np.object)[:,0]
        queries = np.loadtxt(args.img_info, dtype=np.object)[:,1]
        print img_list, queries
        gt_file = args.gt_file
        proposal_dir = args.proposal_dir
        img_dir = args.img_dir
        res_dir = args.res_dir
        if not os.path.exists(res_dir):
            print 'Creating Dir : %s' %res_dir
            os.mkdir(res_dir)
        word_dict = np.load(args.dictFile)
        gen_img_cap_feat(img_list, gt_file, proposal_dir, res_dir, word_dict, queries)
        add_img_gt_label(img_list, gt_file, proposal_dir, res_dir, queries)
        add_img_sen_lang_token(img_list, gt_file, word_dict, proposal_dir, res_dir)
        add_sen_id_label(img_list, res_dir)
        add_img_hw_label(img_list, img_dir, res_dir)
        add_img_reg_label(img_list, res_dir)
        add_img_reg_label_all(img_list, res_dir)


if __name__ == '__main__':
    main()
