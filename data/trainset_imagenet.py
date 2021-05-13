import os
import random
image_path = '/root/data/ImageNetTank'
annotation_path = '/root/data/ImageNetTank_png_seg'

annotations = [xml for xml in os.listdir(annotation_path) if xml.endswith('xml')]
random.shuffle(annotations)
print(len(annotations))
train_size = int(len(annotations)*0.8)
train = annotations[:train_size]
val = annotations[train_size:]

f_train = open('./imagenet_trainval.txt', 'w')
f_val = open('./imagenet_val.txt', 'w')

for f in train:
    f_train.write(f'{f.split(".")[0]}\n')

for f in val:
    f_val.write(f'{f.split(".")[0]}\n')

f_train.close()
f_val.close()


    

