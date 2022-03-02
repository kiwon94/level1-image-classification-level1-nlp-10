import argparse
import os
from importlib import import_module
import model as models
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

def load_model(saved_model, num_classes, device):
    model = models.get_model(args.model, num_classes)
    
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth') # best model path
    model.load_state_dict(torch.load(model_path).module.state_dict()) # 파라미터 load
    
    return model
def kfold_load_model(saved_model, num_classes, device):

    model_module = import_module("model")
    model = model_module.get_model(args.model, num_classes)

    modelfile_list = [modelfile for modelfile in os.listdir(saved_model) if modelfile.endswith('.pth')]
    best_val_loss = np.inf
    best_model_path=''
    for models in modelfile_list:
        _, _, val_acc, val_loss = models.split('_')
        val_loss = float(val_loss.split('.p')[0])
        if val_loss<best_val_loss:
            best_model_path = models
            best_val_loss = val_loss
    model_path = os.path.join(saved_model, best_model_path)
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
    info.to_csv(os.path.join(output_dir, f'output3.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=250, help='input batch size for validing (default: 1000)')
    parser.add_argument("--resize", nargs="+", type=int, default=(128, 96), help='resize size for image when training')
    parser.add_argument('--model', type=str, default='densenet', help='model type (default: BaseModel)')
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
