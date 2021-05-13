# A PyTorch implementation of Faster R-CNN
* 출처 : https://github.com/loolzaaa/faster-rcnn-pytorch
* 목적 : 커스텀 이미지 학습 및 및 학습데이터 증강 추가


## 도커 실행 시
* base image : nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
* Dockerfile 빌드 후 컨테이너 실행
* pretrained_model와 학습데이터 다운로드 후 바로 학습 가능(python3.8)

## Pretrained model 다운로드
1. Download caffe BGR pretrained models ([link](https://drive.google.com/open?id=1n2hWpTEWe3LwfOYq0VUslok-EmdqrMQP)) or PyTorch RGB pretrained models ([link](https://drive.google.com/drive/folders/1P4Q9jtsMB9C47l7imseK5JlTpgMX1pFh?usp=sharing))
2. Put them into `data/pretrained_model/`

### train dataset
* voc 포맷의 annotation파일 준비
* voc_2007 데이터셋의 구조를 그대로 사용(data/VOCdevkit/VOC2007/ImageSet/Main/trainval.txt) <- 수정해서 쓰는게 나음...
* image/annotation 경로는 lib/dataset/voc/pascal_voc.py를 수정 (image_path_at, _load_annotation)

### test/detection dataset
* voc_2007 데이터셋의 구조를 그대로 사용(data/VOCdevkit/VOC2007/ImageSet/Main/test.txt)
* data/images/에 detection할 이미지 저장 후 detect

### example
* train : ```python3.8 run.py train -n resnet50 --cuda```
* test : ```python3.8 run.py test --dataset voc_2007_test -n resnet50 --epoch 20 --cuda```
* detect : ```python3.8 run.py detect --dataset voc_2007_test --net resnet50 --epoch 20 --cuda --vis```


