# OSFI_by_FineTuning
Official implementation for Open-set Face Identification on Few-shot Gallery by Fine-Tuning 

## Requirements
- Pytorch 1.7.1
- Torchvision 0.8.2

## Usage
Download Pretrained Weights:
- <a href="https://drive.google.com/file/d/11TqrfXXdow0SjXbrsiCHEajXInTsuK8o/view?usp=sharing" target="_blank">VGGNet-19</a>
- <a href="https://drive.google.com/file/d/1C534tKYLvEF3e3UwsQscaL7lpaQapMAT/view?usp=sharing" target="_blank">ResNet-50</a>

In config.py change the encoder_config to the directory of the downloaded weights (.chkpt files).

Download Face Datasets:
- <a href="https://drive.google.com/file/d/1PthzzzuufDaJZwZE-YQNQBKs9L9b8TBs/view?usp=sharing" target="_blank">IJB-C</a>
- <a href="https://drive.google.com/file/d/1igvIRI7jVpg01e13HgZ4JibT0BI2GuhS/view?usp=sharing" target="_blank">CASIA-WebFace</a>  

The images are already cropped using <a href="https://github.com/timesler/facenet-pytorch" target="_blank">MTCNN by timesler</a>.  
In config.py change the data_config as below:  
data_config = {
<br />"CASIA": {
<br /><br />"G_data_dir": 
<br />}
}
