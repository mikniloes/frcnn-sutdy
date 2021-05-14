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

### 데이터 증강
![스크린샷 2021-05-13 오전 11 00 09](https://user-images.githubusercontent.com/84064361/118066863-6b566a80-b3da-11eb-9928-29e03312af6e.png)

* python imgaug 라이브러리 사용
* lib/dataset/collate.py
* 포함된 증강 :
  + AverageBlur & MedianBlur
  + AdditiveGaussianNoise
  + AddToHueAndSaturation
  + LinearContrast
  + Multiply
  + Grayscale
  + flip (기존에 포함)
 
### freeze backbone(resnet) layer
* lib/config.py => cfg.RESNET.NUM_FREEZE_BLOCKS = 0~3
* feature extraction layer를 전부 고정할 경우 layer1~3은 학습 X (parameter가 업데이트 되지 않음)
![스크린샷 2021-05-13 오전 10 44 20](https://user-images.githubusercontent.com/84064361/118065603-33e6be80-b3d8-11eb-8deb-f18e99acb010.png)

### checkpoint 파일에서 resnet weight 가져오기
* 학습 모델의 key가 아래 표와 같이 매치됨
|checkpoint(.pth)|resnet50|
|:---|:---|
|RCNN_rpn.RPN_Conv.weight|layer4.2.bn3.num_batches_tracked|
|RCNN_rpn.RPN_Conv.bias|layer4.2.bn3.num_batches_tracked|
|RCNN_rpn.RPN_cls_score.weight|layer4.2.bn3.num_batches_tracked|
|RCNN_rpn.RPN_cls_score.bias|layer4.2.bn3.num_batches_tracked|
|RCNN_rpn.RPN_bbox_pred.weight|layer4.2.bn3.num_batches_tracked|
|RCNN_rpn.RPN_bbox_pred.bias|layer4.2.bn3.num_batches_tracked|
|RCNN_base.0.weight|conv1.weight|
|RCNN_base.1.weight|bn1.weight|
|RCNN_base.1.bias|bn1.bias|
|RCNN_base.1.running_mean|bn1.running_mean|
|RCNN_base.1.running_var|bn1.running_var|
|RCNN_base.1.num_batches_tracked|bn1.num_batches_tracked|
|RCNN_base.4.0.conv1.weight|layer1.0.conv1.weight|
|RCNN_base.4.0.bn1.weight|layer1.0.bn1.weight|
|RCNN_base.4.0.bn1.bias|layer1.0.bn1.bias|
|RCNN_base.4.0.bn1.running_mean|layer1.0.bn1.running_mean|
|RCNN_base.4.0.bn1.running_var|layer1.0.bn1.running_var|
|RCNN_base.4.0.bn1.num_batches_tracked|layer1.0.bn1.num_batches_tracked|
|RCNN_base.4.0.conv2.weight|layer1.0.conv2.weight|
|RCNN_base.4.0.bn2.weight|layer1.0.bn2.weight|
|RCNN_base.4.0.bn2.bias|layer1.0.bn2.bias|
|RCNN_base.4.0.bn2.running_mean|layer1.0.bn2.running_mean|
|RCNN_base.4.0.bn2.running_var|layer1.0.bn2.running_var|
|RCNN_base.4.0.bn2.num_batches_tracked|layer1.0.bn2.num_batches_tracked|
|RCNN_base.4.0.conv3.weight|layer1.0.conv3.weight|
|RCNN_base.4.0.bn3.weight|layer1.0.bn3.weight|
|RCNN_base.4.0.bn3.bias|layer1.0.bn3.bias|
|RCNN_base.4.0.bn3.running_mean|layer1.0.bn3.running_mean|
|RCNN_base.4.0.bn3.running_var|layer1.0.bn3.running_var|
|RCNN_base.4.0.bn3.num_batches_tracked|layer1.0.bn3.num_batches_tracked|
|RCNN_base.4.0.downsample.0.weight|layer1.0.downsample.0.weight|
|RCNN_base.4.0.downsample.1.weight|layer1.0.downsample.1.weight|
|RCNN_base.4.0.downsample.1.bias|layer1.0.downsample.1.bias|
|RCNN_base.4.0.downsample.1.running_mean|layer1.0.downsample.1.running_mean|
|RCNN_base.4.0.downsample.1.running_var|layer1.0.downsample.1.running_var|
|RCNN_base.4.0.downsample.1.num_batches_tracked|layer1.0.downsample.1.num_batches_tracked|
|RCNN_base.4.1.conv1.weight|layer1.1.conv1.weight|
|RCNN_base.4.1.bn1.weight|layer1.1.bn1.weight|
|RCNN_base.4.1.bn1.bias|layer1.1.bn1.bias|
|RCNN_base.4.1.bn1.running_mean|layer1.1.bn1.running_mean|
|RCNN_base.4.1.bn1.running_var|layer1.1.bn1.running_var|
|RCNN_base.4.1.bn1.num_batches_tracked|layer1.1.bn1.num_batches_tracked|
|RCNN_base.4.1.conv2.weight|layer1.1.conv2.weight|
|RCNN_base.4.1.bn2.weight|layer1.1.bn2.weight|
|RCNN_base.4.1.bn2.bias|layer1.1.bn2.bias|
|RCNN_base.4.1.bn2.running_mean|layer1.1.bn2.running_mean|
|RCNN_base.4.1.bn2.running_var|layer1.1.bn2.running_var|
|RCNN_base.4.1.bn2.num_batches_tracked|layer1.1.bn2.num_batches_tracked|
|RCNN_base.4.1.conv3.weight|layer1.1.conv3.weight|
|RCNN_base.4.1.bn3.weight|layer1.1.bn3.weight|
|RCNN_base.4.1.bn3.bias|layer1.1.bn3.bias|
|RCNN_base.4.1.bn3.running_mean|layer1.1.bn3.running_mean|
|RCNN_base.4.1.bn3.running_var|layer1.1.bn3.running_var|
|RCNN_base.4.1.bn3.num_batches_tracked|layer1.1.bn3.num_batches_tracked|
|RCNN_base.4.2.conv1.weight|layer1.2.conv1.weight|
|RCNN_base.4.2.bn1.weight|layer1.2.bn1.weight|
|RCNN_base.4.2.bn1.bias|layer1.2.bn1.bias|
|RCNN_base.4.2.bn1.running_mean|layer1.2.bn1.running_mean|
|RCNN_base.4.2.bn1.running_var|layer1.2.bn1.running_var|
|RCNN_base.4.2.bn1.num_batches_tracked|layer1.2.bn1.num_batches_tracked|
|RCNN_base.4.2.conv2.weight|layer1.2.conv2.weight|
|RCNN_base.4.2.bn2.weight|layer1.2.bn2.weight|
|RCNN_base.4.2.bn2.bias|layer1.2.bn2.bias|
|RCNN_base.4.2.bn2.running_mean|layer1.2.bn2.running_mean|
|RCNN_base.4.2.bn2.running_var|layer1.2.bn2.running_var|
|RCNN_base.4.2.bn2.num_batches_tracked|layer1.2.bn2.num_batches_tracked|
|RCNN_base.4.2.conv3.weight|layer1.2.conv3.weight|
|RCNN_base.4.2.bn3.weight|layer1.2.bn3.weight|
|RCNN_base.4.2.bn3.bias|layer1.2.bn3.bias|
|RCNN_base.4.2.bn3.running_mean|layer1.2.bn3.running_mean|
|RCNN_base.4.2.bn3.running_var|layer1.2.bn3.running_var|
|RCNN_base.4.2.bn3.num_batches_tracked|layer1.2.bn3.num_batches_tracked|
|RCNN_base.5.0.conv1.weight|layer2.0.conv1.weight|
|RCNN_base.5.0.bn1.weight|layer2.0.bn1.weight|
|RCNN_base.5.0.bn1.bias|layer2.0.bn1.bias|
|RCNN_base.5.0.bn1.running_mean|layer2.0.bn1.running_mean|
|RCNN_base.5.0.bn1.running_var|layer2.0.bn1.running_var|
|RCNN_base.5.0.bn1.num_batches_tracked|layer2.0.bn1.num_batches_tracked|
|RCNN_base.5.0.conv2.weight|layer2.0.conv2.weight|
|RCNN_base.5.0.bn2.weight|layer2.0.bn2.weight|
|RCNN_base.5.0.bn2.bias|layer2.0.bn2.bias|
|RCNN_base.5.0.bn2.running_mean|layer2.0.bn2.running_mean|
|RCNN_base.5.0.bn2.running_var|layer2.0.bn2.running_var|
|RCNN_base.5.0.bn2.num_batches_tracked|layer2.0.bn2.num_batches_tracked|
|RCNN_base.5.0.conv3.weight|layer2.0.conv3.weight|
|RCNN_base.5.0.bn3.weight|layer2.0.bn3.weight|
|RCNN_base.5.0.bn3.bias|layer2.0.bn3.bias|
|RCNN_base.5.0.bn3.running_mean|layer2.0.bn3.running_mean|
|RCNN_base.5.0.bn3.running_var|layer2.0.bn3.running_var|
|RCNN_base.5.0.bn3.num_batches_tracked|layer2.0.bn3.num_batches_tracked|
|RCNN_base.5.0.downsample.0.weight|layer2.0.downsample.0.weight|
|RCNN_base.5.0.downsample.1.weight|layer2.0.downsample.1.weight|
|RCNN_base.5.0.downsample.1.bias|layer2.0.downsample.1.bias|
|RCNN_base.5.0.downsample.1.running_mean|layer2.0.downsample.1.running_mean|
|RCNN_base.5.0.downsample.1.running_var|layer2.0.downsample.1.running_var|
|RCNN_base.5.0.downsample.1.num_batches_tracked|layer2.0.downsample.1.num_batches_tracked|
|RCNN_base.5.1.conv1.weight|layer2.1.conv1.weight|
|RCNN_base.5.1.bn1.weight|layer2.1.bn1.weight|
|RCNN_base.5.1.bn1.bias|layer2.1.bn1.bias|
|RCNN_base.5.1.bn1.running_mean|layer2.1.bn1.running_mean|
|RCNN_base.5.1.bn1.running_var|layer2.1.bn1.running_var|
|RCNN_base.5.1.bn1.num_batches_tracked|layer2.1.bn1.num_batches_tracked|
|RCNN_base.5.1.conv2.weight|layer2.1.conv2.weight|
|RCNN_base.5.1.bn2.weight|layer2.1.bn2.weight|
|RCNN_base.5.1.bn2.bias|layer2.1.bn2.bias|
|RCNN_base.5.1.bn2.running_mean|layer2.1.bn2.running_mean|
|RCNN_base.5.1.bn2.running_var|layer2.1.bn2.running_var|
|RCNN_base.5.1.bn2.num_batches_tracked|layer2.1.bn2.num_batches_tracked|
|RCNN_base.5.1.conv3.weight|layer2.1.conv3.weight|
|RCNN_base.5.1.bn3.weight|layer2.1.bn3.weight|
|RCNN_base.5.1.bn3.bias|layer2.1.bn3.bias|
|RCNN_base.5.1.bn3.running_mean|layer2.1.bn3.running_mean|
|RCNN_base.5.1.bn3.running_var|layer2.1.bn3.running_var|
|RCNN_base.5.1.bn3.num_batches_tracked|layer2.1.bn3.num_batches_tracked|
|RCNN_base.5.2.conv1.weight|layer2.2.conv1.weight|
|RCNN_base.5.2.bn1.weight|layer2.2.bn1.weight|
|RCNN_base.5.2.bn1.bias|layer2.2.bn1.bias|
|RCNN_base.5.2.bn1.running_mean|layer2.2.bn1.running_mean|
|RCNN_base.5.2.bn1.running_var|layer2.2.bn1.running_var|
|RCNN_base.5.2.bn1.num_batches_tracked|layer2.2.bn1.num_batches_tracked|
|RCNN_base.5.2.conv2.weight|layer2.2.conv2.weight|
|RCNN_base.5.2.bn2.weight|layer2.2.bn2.weight|
|RCNN_base.5.2.bn2.bias|layer2.2.bn2.bias|
|RCNN_base.5.2.bn2.running_mean|layer2.2.bn2.running_mean|
|RCNN_base.5.2.bn2.running_var|layer2.2.bn2.running_var|
|RCNN_base.5.2.bn2.num_batches_tracked|layer2.2.bn2.num_batches_tracked|
|RCNN_base.5.2.conv3.weight|layer2.2.conv3.weight|
|RCNN_base.5.2.bn3.weight|layer2.2.bn3.weight|
|RCNN_base.5.2.bn3.bias|layer2.2.bn3.bias|
|RCNN_base.5.2.bn3.running_mean|layer2.2.bn3.running_mean|
|RCNN_base.5.2.bn3.running_var|layer2.2.bn3.running_var|
|RCNN_base.5.2.bn3.num_batches_tracked|layer2.2.bn3.num_batches_tracked|
|RCNN_base.5.3.conv1.weight|layer2.3.conv1.weight|
|RCNN_base.5.3.bn1.weight|layer2.3.bn1.weight|
|RCNN_base.5.3.bn1.bias|layer2.3.bn1.bias|
|RCNN_base.5.3.bn1.running_mean|layer2.3.bn1.running_mean|
|RCNN_base.5.3.bn1.running_var|layer2.3.bn1.running_var|
|RCNN_base.5.3.bn1.num_batches_tracked|layer2.3.bn1.num_batches_tracked|
|RCNN_base.5.3.conv2.weight|layer2.3.conv2.weight|
|RCNN_base.5.3.bn2.weight|layer2.3.bn2.weight|
|RCNN_base.5.3.bn2.bias|layer2.3.bn2.bias|
|RCNN_base.5.3.bn2.running_mean|layer2.3.bn2.running_mean|
|RCNN_base.5.3.bn2.running_var|layer2.3.bn2.running_var|
|RCNN_base.5.3.bn2.num_batches_tracked|layer2.3.bn2.num_batches_tracked|
|RCNN_base.5.3.conv3.weight|layer2.3.conv3.weight|
|RCNN_base.5.3.bn3.weight|layer2.3.bn3.weight|
|RCNN_base.5.3.bn3.bias|layer2.3.bn3.bias|
|RCNN_base.5.3.bn3.running_mean|layer2.3.bn3.running_mean|
|RCNN_base.5.3.bn3.running_var|layer2.3.bn3.running_var|
|RCNN_base.5.3.bn3.num_batches_tracked|layer2.3.bn3.num_batches_tracked|
|RCNN_base.6.0.conv1.weight|layer3.0.conv1.weight|
|RCNN_base.6.0.bn1.weight|layer3.0.bn1.weight|
|RCNN_base.6.0.bn1.bias|layer3.0.bn1.bias|
|RCNN_base.6.0.bn1.running_mean|layer3.0.bn1.running_mean|
|RCNN_base.6.0.bn1.running_var|layer3.0.bn1.running_var|
|RCNN_base.6.0.bn1.num_batches_tracked|layer3.0.bn1.num_batches_tracked|
|RCNN_base.6.0.conv2.weight|layer3.0.conv2.weight|
|RCNN_base.6.0.bn2.weight|layer3.0.bn2.weight|
|RCNN_base.6.0.bn2.bias|layer3.0.bn2.bias|
|RCNN_base.6.0.bn2.running_mean|layer3.0.bn2.running_mean|
|RCNN_base.6.0.bn2.running_var|layer3.0.bn2.running_var|
|RCNN_base.6.0.bn2.num_batches_tracked|layer3.0.bn2.num_batches_tracked|
|RCNN_base.6.0.conv3.weight|layer3.0.conv3.weight|
|RCNN_base.6.0.bn3.weight|layer3.0.bn3.weight|
|RCNN_base.6.0.bn3.bias|layer3.0.bn3.bias|
|RCNN_base.6.0.bn3.running_mean|layer3.0.bn3.running_mean|
|RCNN_base.6.0.bn3.running_var|layer3.0.bn3.running_var|
|RCNN_base.6.0.bn3.num_batches_tracked|layer3.0.bn3.num_batches_tracked|
|RCNN_base.6.0.downsample.0.weight|layer3.0.downsample.0.weight|
|RCNN_base.6.0.downsample.1.weight|layer3.0.downsample.1.weight|
|RCNN_base.6.0.downsample.1.bias|layer3.0.downsample.1.bias|
|RCNN_base.6.0.downsample.1.running_mean|layer3.0.downsample.1.running_mean|
|RCNN_base.6.0.downsample.1.running_var|layer3.0.downsample.1.running_var|
|RCNN_base.6.0.downsample.1.num_batches_tracked|layer3.0.downsample.1.num_batches_tracked|
|RCNN_base.6.1.conv1.weight|layer3.1.conv1.weight|
|RCNN_base.6.1.bn1.weight|layer3.1.bn1.weight|
|RCNN_base.6.1.bn1.bias|layer3.1.bn1.bias|
|RCNN_base.6.1.bn1.running_mean|layer3.1.bn1.running_mean|
|RCNN_base.6.1.bn1.running_var|layer3.1.bn1.running_var|
|RCNN_base.6.1.bn1.num_batches_tracked|layer3.1.bn1.num_batches_tracked|
|RCNN_base.6.1.conv2.weight|layer3.1.conv2.weight|
|RCNN_base.6.1.bn2.weight|layer3.1.bn2.weight|
|RCNN_base.6.1.bn2.bias|layer3.1.bn2.bias|
|RCNN_base.6.1.bn2.running_mean|layer3.1.bn2.running_mean|
|RCNN_base.6.1.bn2.running_var|layer3.1.bn2.running_var|
|RCNN_base.6.1.bn2.num_batches_tracked|layer3.1.bn2.num_batches_tracked|
|RCNN_base.6.1.conv3.weight|layer3.1.conv3.weight|
|RCNN_base.6.1.bn3.weight|layer3.1.bn3.weight|
|RCNN_base.6.1.bn3.bias|layer3.1.bn3.bias|
|RCNN_base.6.1.bn3.running_mean|layer3.1.bn3.running_mean|
|RCNN_base.6.1.bn3.running_var|layer3.1.bn3.running_var|
|RCNN_base.6.1.bn3.num_batches_tracked|layer3.1.bn3.num_batches_tracked|
|RCNN_base.6.2.conv1.weight|layer3.2.conv1.weight|
|RCNN_base.6.2.bn1.weight|layer3.2.bn1.weight|
|RCNN_base.6.2.bn1.bias|layer3.2.bn1.bias|
|RCNN_base.6.2.bn1.running_mean|layer3.2.bn1.running_mean|
|RCNN_base.6.2.bn1.running_var|layer3.2.bn1.running_var|
|RCNN_base.6.2.bn1.num_batches_tracked|layer3.2.bn1.num_batches_tracked|
|RCNN_base.6.2.conv2.weight|layer3.2.conv2.weight|
|RCNN_base.6.2.bn2.weight|layer3.2.bn2.weight|
|RCNN_base.6.2.bn2.bias|layer3.2.bn2.bias|
|RCNN_base.6.2.bn2.running_mean|layer3.2.bn2.running_mean|
|RCNN_base.6.2.bn2.running_var|layer3.2.bn2.running_var|
|RCNN_base.6.2.bn2.num_batches_tracked|layer3.2.bn2.num_batches_tracked|
|RCNN_base.6.2.conv3.weight|layer3.2.conv3.weight|
|RCNN_base.6.2.bn3.weight|layer3.2.bn3.weight|
|RCNN_base.6.2.bn3.bias|layer3.2.bn3.bias|
|RCNN_base.6.2.bn3.running_mean|layer3.2.bn3.running_mean|
|RCNN_base.6.2.bn3.running_var|layer3.2.bn3.running_var|
|RCNN_base.6.2.bn3.num_batches_tracked|layer3.2.bn3.num_batches_tracked|
|RCNN_base.6.3.conv1.weight|layer3.3.conv1.weight|
|RCNN_base.6.3.bn1.weight|layer3.3.bn1.weight|
|RCNN_base.6.3.bn1.bias|layer3.3.bn1.bias|
|RCNN_base.6.3.bn1.running_mean|layer3.3.bn1.running_mean|
|RCNN_base.6.3.bn1.running_var|layer3.3.bn1.running_var|
|RCNN_base.6.3.bn1.num_batches_tracked|layer3.3.bn1.num_batches_tracked|
|RCNN_base.6.3.conv2.weight|layer3.3.conv2.weight|
|RCNN_base.6.3.bn2.weight|layer3.3.bn2.weight|
|RCNN_base.6.3.bn2.bias|layer3.3.bn2.bias|
|RCNN_base.6.3.bn2.running_mean|layer3.3.bn2.running_mean|
|RCNN_base.6.3.bn2.running_var|layer3.3.bn2.running_var|
|RCNN_base.6.3.bn2.num_batches_tracked|layer3.3.bn2.num_batches_tracked|
|RCNN_base.6.3.conv3.weight|layer3.3.conv3.weight|
|RCNN_base.6.3.bn3.weight|layer3.3.bn3.weight|
|RCNN_base.6.3.bn3.bias|layer3.3.bn3.bias|
|RCNN_base.6.3.bn3.running_mean|layer3.3.bn3.running_mean|
|RCNN_base.6.3.bn3.running_var|layer3.3.bn3.running_var|
|RCNN_base.6.3.bn3.num_batches_tracked|layer3.3.bn3.num_batches_tracked|
|RCNN_base.6.4.conv1.weight|layer3.4.conv1.weight|
|RCNN_base.6.4.bn1.weight|layer3.4.bn1.weight|
|RCNN_base.6.4.bn1.bias|layer3.4.bn1.bias|
|RCNN_base.6.4.bn1.running_mean|layer3.4.bn1.running_mean|
|RCNN_base.6.4.bn1.running_var|layer3.4.bn1.running_var|
|RCNN_base.6.4.bn1.num_batches_tracked|layer3.4.bn1.num_batches_tracked|
|RCNN_base.6.4.conv2.weight|layer3.4.conv2.weight|
|RCNN_base.6.4.bn2.weight|layer3.4.bn2.weight|
|RCNN_base.6.4.bn2.bias|layer3.4.bn2.bias|
|RCNN_base.6.4.bn2.running_mean|layer3.4.bn2.running_mean|
|RCNN_base.6.4.bn2.running_var|layer3.4.bn2.running_var|
|RCNN_base.6.4.bn2.num_batches_tracked|layer3.4.bn2.num_batches_tracked|
|RCNN_base.6.4.conv3.weight|layer3.4.conv3.weight|
|RCNN_base.6.4.bn3.weight|layer3.4.bn3.weight|
|RCNN_base.6.4.bn3.bias|layer3.4.bn3.bias|
|RCNN_base.6.4.bn3.running_mean|layer3.4.bn3.running_mean|
|RCNN_base.6.4.bn3.running_var|layer3.4.bn3.running_var|
|RCNN_base.6.4.bn3.num_batches_tracked|layer3.4.bn3.num_batches_tracked|
|RCNN_base.6.5.conv1.weight|layer3.5.conv1.weight|
|RCNN_base.6.5.bn1.weight|layer3.5.bn1.weight|
|RCNN_base.6.5.bn1.bias|layer3.5.bn1.bias|
|RCNN_base.6.5.bn1.running_mean|layer3.5.bn1.running_mean|
|RCNN_base.6.5.bn1.running_var|layer3.5.bn1.running_var|
|RCNN_base.6.5.bn1.num_batches_tracked|layer3.5.bn1.num_batches_tracked|
|RCNN_base.6.5.conv2.weight|layer3.5.conv2.weight|
|RCNN_base.6.5.bn2.weight|layer3.5.bn2.weight|
|RCNN_base.6.5.bn2.bias|layer3.5.bn2.bias|
|RCNN_base.6.5.bn2.running_mean|layer3.5.bn2.running_mean|
|RCNN_base.6.5.bn2.running_var|layer3.5.bn2.running_var|
|RCNN_base.6.5.bn2.num_batches_tracked|layer3.5.bn2.num_batches_tracked|
|RCNN_base.6.5.conv3.weight|layer3.5.conv3.weight|
|RCNN_base.6.5.bn3.weight|layer3.5.bn3.weight|
|RCNN_base.6.5.bn3.bias|layer3.5.bn3.bias|
|RCNN_base.6.5.bn3.running_mean|layer3.5.bn3.running_mean|
|RCNN_base.6.5.bn3.running_var|layer3.5.bn3.running_var|
|RCNN_base.6.5.bn3.num_batches_tracked|layer3.5.bn3.num_batches_tracked|
|RCNN_top.0.0.conv1.weight|layer4.0.conv1.weight|
|RCNN_top.0.0.bn1.weight|layer4.0.bn1.weight|
|RCNN_top.0.0.bn1.bias|layer4.0.bn1.bias|
|RCNN_top.0.0.bn1.running_mean|layer4.0.bn1.running_mean|
|RCNN_top.0.0.bn1.running_var|layer4.0.bn1.running_var|
|RCNN_top.0.0.bn1.num_batches_tracked|layer4.0.bn1.num_batches_tracked|
|RCNN_top.0.0.conv2.weight|layer4.0.conv2.weight|
|RCNN_top.0.0.bn2.weight|layer4.0.bn2.weight|
|RCNN_top.0.0.bn2.bias|layer4.0.bn2.bias|
|RCNN_top.0.0.bn2.running_mean|layer4.0.bn2.running_mean|
|RCNN_top.0.0.bn2.running_var|layer4.0.bn2.running_var|
|RCNN_top.0.0.bn2.num_batches_tracked|layer4.0.bn2.num_batches_tracked|
|RCNN_top.0.0.conv3.weight|layer4.0.conv3.weight|
|RCNN_top.0.0.bn3.weight|layer4.0.bn3.weight|
|RCNN_top.0.0.bn3.bias|layer4.0.bn3.bias|
|RCNN_top.0.0.bn3.running_mean|layer4.0.bn3.running_mean|
|RCNN_top.0.0.bn3.running_var|layer4.0.bn3.running_var|
|RCNN_top.0.0.bn3.num_batches_tracked|layer4.0.bn3.num_batches_tracked|
|RCNN_top.0.0.downsample.0.weight|layer4.0.downsample.0.weight|
|RCNN_top.0.0.downsample.1.weight|layer4.0.downsample.1.weight|
|RCNN_top.0.0.downsample.1.bias|layer4.0.downsample.1.bias|
|RCNN_top.0.0.downsample.1.running_mean|layer4.0.downsample.1.running_mean|
|RCNN_top.0.0.downsample.1.running_var|layer4.0.downsample.1.running_var|
|RCNN_top.0.0.downsample.1.num_batches_tracked|layer4.0.downsample.1.num_batches_tracked|
|RCNN_top.0.1.conv1.weight|layer4.1.conv1.weight|
|RCNN_top.0.1.bn1.weight|layer4.1.bn1.weight|
|RCNN_top.0.1.bn1.bias|layer4.1.bn1.bias|
|RCNN_top.0.1.bn1.running_mean|layer4.1.bn1.running_mean|
|RCNN_top.0.1.bn1.running_var|layer4.1.bn1.running_var|
|RCNN_top.0.1.bn1.num_batches_tracked|layer4.1.bn1.num_batches_tracked|
|RCNN_top.0.1.conv2.weight|layer4.1.conv2.weight|
|RCNN_top.0.1.bn2.weight|layer4.1.bn2.weight|
|RCNN_top.0.1.bn2.bias|layer4.1.bn2.bias|
|RCNN_top.0.1.bn2.running_mean|layer4.1.bn2.running_mean|
|RCNN_top.0.1.bn2.running_var|layer4.1.bn2.running_var|
|RCNN_top.0.1.bn2.num_batches_tracked|layer4.1.bn2.num_batches_tracked|
|RCNN_top.0.1.conv3.weight|layer4.1.conv3.weight|
|RCNN_top.0.1.bn3.weight|layer4.1.bn3.weight|
|RCNN_top.0.1.bn3.bias|layer4.1.bn3.bias|
|RCNN_top.0.1.bn3.running_mean|layer4.1.bn3.running_mean|
|RCNN_top.0.1.bn3.running_var|layer4.1.bn3.running_var|
|RCNN_top.0.1.bn3.num_batches_tracked|layer4.1.bn3.num_batches_tracked|
|RCNN_top.0.2.conv1.weight|layer4.2.conv1.weight|
|RCNN_top.0.2.bn1.weight|layer4.2.bn1.weight|
|RCNN_top.0.2.bn1.bias|layer4.2.bn1.bias|
|RCNN_top.0.2.bn1.running_mean|layer4.2.bn1.running_mean|
|RCNN_top.0.2.bn1.running_var|layer4.2.bn1.running_var|
|RCNN_top.0.2.bn1.num_batches_tracked|layer4.2.bn1.num_batches_tracked|
|RCNN_top.0.2.conv2.weight|layer4.2.conv2.weight|
|RCNN_top.0.2.bn2.weight|layer4.2.bn2.weight|
|RCNN_top.0.2.bn2.bias|layer4.2.bn2.bias|
|RCNN_top.0.2.bn2.running_mean|layer4.2.bn2.running_mean|
|RCNN_top.0.2.bn2.running_var|layer4.2.bn2.running_var|
|RCNN_top.0.2.bn2.num_batches_tracked|layer4.2.bn2.num_batches_tracked|
|RCNN_top.0.2.conv3.weight|layer4.2.conv3.weight|
|RCNN_top.0.2.bn3.weight|layer4.2.bn3.weight|
|RCNN_top.0.2.bn3.bias|layer4.2.bn3.bias|
|RCNN_top.0.2.bn3.running_mean|layer4.2.bn3.running_mean|
|RCNN_top.0.2.bn3.running_var|layer4.2.bn3.running_var|
|RCNN_top.0.2.bn3.num_batches_tracked|layer4.2.bn3.num_batches_tracked|
|RCNN_cls_score.weight|layer4.2.bn3.num_batches_tracked|
|RCNN_cls_score.bias|layer4.2.bn3.num_batches_tracked|
|RCNN_bbox_pred.weight|layer4.2.bn3.num_batches_tracked|
|RCNN_bbox_pred.bias|layer4.2.bn3.num_batches_tracked|

