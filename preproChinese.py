# -*- coding: UTF-8 -*-  
from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
import jieba


def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    count = 0
    data = []
    for annotation in caption_data:
        for i in range(len(annotation['caption'])):
            caption = annotation['caption'][i]
            caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
            caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
            caption = " ".join(caption.split())  # replace multiple spaces
            seg_list = jieba.lcut(caption,cut_all = False)
            count += 1
            print count
            if (len(seg_list) > max_length):
                continue
            annotation1 = {}
            annotation1['file_name'] = os.path.join(image_dir, annotation['image_id']) 
            annotation1['caption'] = caption
            data += [annotation1]
            
    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    
    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    
    for i, caption in enumerate(annotations['caption']):
        words = jieba.lcut(caption,cut_all = False) # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
       
        print i
        if len(words) > max_len:
            max_len = len(words)

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    
    for i, caption in enumerate(annotations['caption']):
        print i
        words = jieba.lcut(caption,cut_all = False) # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_filename = {}
    idx = 0
    file_names = annotations['file_name']
    for file_name in file_names:
        if not file_name in id_to_filename:
            id_to_filename[file_name] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_filename


def _build_image_idxs(annotations, id_to_filename):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    file_names = annotations['file_name']
    for i, file_name in enumerate(file_names):
        image_idxs[i] = id_to_filename[file_name]
    return image_idxs


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 20
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path 
    vgg_model_path = './ai.challenger/data/imagenet-vgg-verydeep-19.mat'

    caption_file_train = './ai.challenger/data/annotations/caption_train_annotations_20170902.json'
    caption_file_val = './ai.challenger/data/annotations/caption_validation_annotations_20170910.json'
    image_dir_train = './ai.challenger/image/train_images_2017_resized/'
    image_dir_val = './ai.challenger/image/validation_images_2017_resized/'
   
    
    word_to_idx = load_pickle('./data/%s/word_to_idx.pkl' % 'train')
    
    # about 210000 images and 1050000 captions for train dataset
    #train_dataset = _process_caption_data(caption_file,
    #                                      image_dir_train,
    #                                      max_length)

    # about 30000 images and 150000 captions
    val_dataset = _process_caption_data(caption_file_val,
                                        image_dir_val,
                                        max_length)

    # about 4000 images and 20000 captions for val / test dataset
    val_cutoff = int(0.1 * len(val_dataset))
    test_cutoff = int(0.2 * len(val_dataset))
    print 'Finished processing caption data'

    #save_pickle(train_dataset, 'data/train/train.annotations.pkl')
    save_pickle(val_dataset[:val_cutoff], 'data/val/val.annotations.pkl')
    save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')

    for split in [ 'val', 'test']:
        annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))
        if split == 'train':
            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)
            #word_to_idx = load_pickle('./data/%s/word_to_idx.pkl' % split)
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))
        #caption = load_pickle('./data/%s/%s.captions.pkl' % (split, split))
        file_names, id_to_filename = _build_file_names(annotations)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))
        

        print "Finished loading"
        image_idxs = _build_image_idxs(annotations, id_to_filename)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, file_name in zip(annotations['caption'], annotations['file_name']):
            if not file_name in image_ids:
                image_ids[file_name] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption + ' .')
        save_pickle(feature_to_captions, './data/%s/%s.references.pkl' % (split, split))
        print "Finished building %s caption dataset" %split

    # extract conv5_3 feature vectors
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for split in ['train', 'val', 'test']:
            anno_path = './data/%s/%s.annotations.pkl' % (split, split)
            save_path = './data/%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = list(annotations['file_name'].unique())
            n_examples = len(image_path)

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()