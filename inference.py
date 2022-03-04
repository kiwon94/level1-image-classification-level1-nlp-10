import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import gc

from dataset import TestDataset, MaskBaseDataset

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')

# torch.save(model.module.state_dict(), f"{save_dir}/{i:02}_{epoch:03}_{val_acc:4.2%}_{val_loss:4.2}.pt")
def kfold_load_model(saved_model, num_classes, device):

    model_module = import_module("model")
    model = model_module.get_model(args.model, num_classes)

    modelfile_list = [modelfile for modelfile in os.listdir(saved_model) if modelfile.startswith('0')]
    best_val_loss = np.inf
    best_model_path=''
    for models in modelfile_list:
        _, _, _, _, val_loss = models.split('_')
        val_loss = float(val_loss.split('.p')[0])
        if val_loss<best_val_loss:
            best_model_path = models
            best_val_loss = val_loss
    model_path = os.path.join(saved_model, best_model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_model(saved_model, num_classes, device):

    model_module = import_module("model")
    model = model_module.get_model(args.model, num_classes)
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    gc.collect()
    torch.cuda.empty_cache()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    if args.KfoldCV == True:
        model = kfold_load_model(model_dir, num_classes, device).to(device)
    else:
        model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output12.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument("--resize", nargs="+", type=int, default=(224, 224), help='resize size for image when training')
    parser.add_argument('--model', type=str, default='efficientnet_b4', help='model type (default: BaseModel)')
    parser.add_argument('--KfoldCV', type=str2bool, default=True, help='using KfoldCV, default is True')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
