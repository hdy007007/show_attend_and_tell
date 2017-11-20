from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
import os
import pickle

def main():
    
    val_data = load_coco_data(data_path='./data', split='val')
    
    with open(os.path.join('data/train/word_to_idx.pkl'), 'rb') as f:
        word_to_idx = pickle.load(f)
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                             dim_hidden=1024, n_time_step=21, prev2out=True, 
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
 
            
    solver = CaptioningSolver(model, val_data, n_epochs=100, batch_size=128, update_rule='adam',
                             learning_rate=0.0012, print_every=100, save_every=5, image_path='./image/',
                            pretrained_model = 'train_batch/model0.001/model.ckpt-30', model_path='train_batch/model0.002/', test_model=None,
                             print_bleu=True, log_path='train_batch/log/')
        
    solver.train()
 

if __name__ == "__main__":
    main()