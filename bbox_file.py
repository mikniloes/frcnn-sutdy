import os
import random

path = '/root/data/virtual_Data/unreal_5000_default/train'
imgs = [i for i in os.listdir(path) if i.endswith('Object.png')]
random.shuffle(imgs)
imgs = imgs[:50]

xml_path = '/root/faster-rcnn-pytorch/data/VOCdevkit/VOC2007/Annotations'
xml = [f'{img.split(".")[0]}.xml' for img in imgs]

save_path = '/root/data/bbox_check'

for img in imgs:
    img_file = os.path.join(path, img)
    xml_file = os.path.join(xml_path, f'{img.split(".")[0]}.xml')

    command = f"cp {img_file} {save_path}"
    os.system(command)
    print(command)
    command = f"cp {xml_file} {save_path}"
    os.system(command) 
    print(command)
   
