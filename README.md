# OSFI-by-FineTuning
Official implementation for Open-set Face Identification on Few-shot Gallery by Fine-Tuning. <br/>
Published on 2022 International Conference on Pattern Recognition ([Paper](https://arxiv.org/abs/2301.01922)). <br/>
<img src = "https://user-images.githubusercontent.com/60504706/185028026-6d9abe0f-7bfc-4b6f-a369-467a2418dbf6.png" width="65%" height="65%">

## ICPR2022 Presentation
https://user-images.githubusercontent.com/60504706/185032003-813d3b5e-53c7-405f-bae1-017364d5615d.mp4







## Requirements
- Pytorch 1.7.1
- Torchvision 0.8.2

## Setup
Download Pretrained Weights:
- <a href="https://drive.google.com/file/d/1lSOQiNIT1in0bDv2JlpJq9FIxMVFlfmf/view?usp=share_link" target="_blank">VGGNet-19</a>
- <a href="https://drive.google.com/file/d/1Y4xrYKl8GafGDvmT5FBUmnYnhxE0LWbX/view?usp=share_link" target="_blank">ResNet-50</a>

Download Dataset:
- <a href="https://drive.google.com/file/d/1j7fY_fnzESuyIel_f6Z8rZ9ehRuuX0ei/view?usp=share_link" target="_blank"> CASIA-WebFace</a>

Above is the CASIA-WebFace dataset. The images are already cropped using <a href="https://github.com/timesler/facenet-pytorch" target="_blank">MTCNN by timesler</a>.  
To use your own face dataset, you can simply change the ```data_config``` in ```config.py```.  
The face dataset must have the structure ```ROOT/SUBJECT_NAME/image.jpg```.  

After downloading, change the ```dataset_config``` and ```encoder_config``` in ```config.py``` accordingly.

## Usage
After the setup is done, simply run:  
```python main.py --dataset='CASIA' --encoder='VGG19' --classifier_init='WI' --finetune_layers='BN'```  

For further information on the arguments, please refer to our paper.
