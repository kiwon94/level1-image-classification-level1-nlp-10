# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- python train.py --augmentation CustomAugmentation --resize 256 192 --model densenet --optimizer AdamW --lr 0.00009 --criterion focal --lr_decay_step 1 --pretrained True --early_stop 1 --LR_scheduler StepLR --steplr_gamma 0.1
-  "transform": "Compose(\n    
    -  Resize(size=[256, 192], 
    -  interpolation=PIL.Image.BILINEAR)\n    
    -  RandomHorizontalFlip(p=0.5)\n    
    -  ToTensor()\n    
    -  Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))\n)" 
### Inference
-  python inference.py --resize 256 192 --model densenet --model_dir ./model/exp_number

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
