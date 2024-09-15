from data_processor import write_lines
import os 
import random
data_dir ="/content/drive/MyDrive/DS_store/DS_train/Det_train"
img_dir ="/content/drive/MyDrive/DS_store/DS_train/Det_train/test"
fn_list = os.listdir(img_dir)
random.shuffle(fn_list)
test_list = fn_list[:]
write_lines(os.path.join(data_dir,"test.list"), test_list)
