import cache_coco_kp
import os
import torch

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main ():
    if len(os.listdir('cache/coco_train/images')) == 0:
        cache_coco_kp.cache_train_data()
    else:    
        print("Cached Data Already Exists, Proceeding To Training")
    

if __name__ == '__main__':
    main()