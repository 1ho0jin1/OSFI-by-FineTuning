# OSFI_by_FineTuning
Official implementation for Open-set Face Identification on Few-shot Gallery by Fine-Tuning 

## Requirements
- Pytorch 1.7.1
- Torchvision 0.8.2

## Setup
Download Pretrained Weights:
- <a href="https://drive.google.com/file/d/11TqrfXXdow0SjXbrsiCHEajXInTsuK8o/view?usp=sharing" target="_blank">VGGNet-19</a>
- <a href="https://drive.google.com/file/d/1C534tKYLvEF3e3UwsQscaL7lpaQapMAT/view?usp=sharing" target="_blank">ResNet-50</a>

In ```config.py``` change the ```encoder_config``` to the directory of the downloaded weights (.chkpt files).

Download Face Datasets:
- <a href="https://drive.google.com/file/d/1PthzzzuufDaJZwZE-YQNQBKs9L9b8TBs/view?usp=sharing" target="_blank">IJBC</a>
- <a href="https://drive.google.com/file/d/1igvIRI7jVpg01e13HgZ4JibT0BI2GuhS/view?usp=sharing" target="_blank">CASIA_clean</a>  

The images are already cropped using <a href="https://github.com/timesler/facenet-pytorch" target="_blank">MTCNN by timesler</a>.  
In ```config.py``` change the ```data_config```:
- ```known_pkl``` and ```unknown_pkl``` refers to the .pkl files that contain the known and unknown identities
  - set these to the .pkl file in the dataset accordingly
- ```G_data_dir```, ```K_data_dir``` and ```U_data_dir``` refers to the image directory for Gallery, Known and Unknown set respectively.
  - For CASIA-WebFace, set all three to be **your_directory / CASIA_clean / cropped_images**
  - For IJBC, set ```G_data_dir``` to **your_directory / IJBC / images_organized**
  - ```K_data_dir``` and ```U_data_dir``` to **your_directory / IJBC / frames_organized**  

Please refer to the given ```config.py``` format.  

## Usage
After the setup is done, simply run:  
```python main.py --dataset='IJBC' --encoder='VGG19' --classifier_init='WI' --finetune_layers='BN'```  

For further information on the arguments, please refer to our paper.
