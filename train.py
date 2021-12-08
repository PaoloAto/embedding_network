import cache_coco_kp
import os

def main ():
    if len(os.listdir('cache/coco_train/images')) == 0:
        cache_coco_kp.cache_train_data()
    else:    
        print("Cached Data Already Exists, Proceeding To Training")
    
    

        

if __name__ == '__main__':
    main()