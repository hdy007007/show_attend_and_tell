from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
import os
import pickle
import hickle
from core.utils import *

def main():
    
    test1_data = {}
    test1_data['features'] = hickle.load('./data/test1/test1.features.hkl')
    test1_data['file_names'] = load_pickle('./data/test1/test1.file.names.pkl')
    print "Fnished loading..."
    with open(os.path.join('data/train/word_to_idx.pkl'), 'rb') as f:
        word_to_idx = pickle.load(f)
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                             dim_hidden=1024, n_time_step=21, prev2out=True, 
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
 
            
    solver = CaptioningSolver(model, test1_data, n_epochs=100, batch_size=50, update_rule='adam',
                             learning_rate=0.001, print_every=100, save_every=5, image_path='./image/',
                             test_model='./train_batch/model0.001/model.ckpt-30',
                             print_bleu=True, log_path='train_batch/log_test/')
        
    solver.test(test1_data, split='test1', attention_visualization=False, save_sampled_captions=True)
 

if __name__ == "__main__":
    main()