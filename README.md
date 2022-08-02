# OSFI-by-FineTuning
Official implementation for Open-set Face Identification on Few-shot Gallery by Fine-Tuning (ICPR 2022)
<img src = "https://github.com/1ho0jin1/OSFI-by-FineTuning/blob/main/fig_pipeline.png" width="75%" height="75%">

## Requirements
- Pytorch 1.7.1
- Torchvision 0.8.2

## Setup
Download Pretrained Weights:
- <a href="https://drive.google.com/file/d/11TqrfXXdow0SjXbrsiCHEajXInTsuK8o/view?usp=sharing" target="_blank">VGGNet-19</a>
- <a href="https://drive.google.com/file/d/1C534tKYLvEF3e3UwsQscaL7lpaQapMAT/view?usp=sharing" target="_blank">ResNet-50</a>

Download Dataset:
- <a href="https://drive.google.com/file/d/1ByDgiUBTwx9Y2A1pnt8b3nYdIk_PLBfT/view?usp=sharing" target="_blank"> CASIA_subset</a>

Above is a randomly chosen small subset of the CASIA-WebFace.  
The images are already cropped using <a href="https://github.com/timesler/facenet-pytorch" target="_blank">MTCNN by timesler</a>.  
To use your own face dataset, you can simply change the ```data_config``` in ```config.py```.  
The face dataset must have the structure ```ROOT/SUBJECT_NAME/image.jpg```.  

After downloading, change the ```dataset_config``` and ```encoder_config``` in ```config.py``` accordingly.

## Usage
After the setup is done, simply run:  
```python main.py --dataset='CASIA' --encoder='VGG19' --classifier_init='WI' --finetune_layers='BN'```  

For further information on the arguments, please refer to our paper.  
If you want to reproduce the complete results of our paper, please contact: 2014142100@yonsei.ac.kr.
