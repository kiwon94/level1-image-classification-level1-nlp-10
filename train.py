import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from sched import scheduler
from pytorchtools import EarlyStopping
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
import pandas
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from dataset import MaskBaseDataset
from loss import create_criterion

def seed_everything(seed): # seed 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer): # learning rate 불러오기
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
def get_model(device, num_classes=18): #model 불러오기
    
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel

    if args.pretrained=='True' and 'densenet' in args.model: 
        model = model_module(
            pretrained = True,
        ).to(device)

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes) #densenet의 마지막 layer output 차원 변경

    elif args.pretrained=='True' and 'resnet' in args.model:
        model = model_module(
            pretrained = True,
        ).to(device)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) #resnet의 마지막 layer output 차원 변경


    elif args.pretrained=='True' and 'vgg' in args.model:
        model = model_module(
            pretrained = True,
        ).to(device)

        model.classifier[6] = nn.Linear(4096, num_classes)

    else:
        model = model_module(
            num_classes=num_classes
    ).to(device)

    model = torch.nn.DataParallel(model) # 병렬처리
    
    return model
def get_scheduler(optimizer):
    # -- Scheduler
    if args.LR_scheduler == 'GradualWarmupScheduler' :
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)

    elif args.LR_scheduler == 'StepLR' :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.steplr_gamma)

    return scheduler

def get_dataset():
     # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskSplitByProfileDataset
    dataset = dataset_module( # MaskSplitByProfileDataset 생성
        data_dir=data_dir, # /opt/ml/input/data/train/images
    )

    return dataset
    
def get_transform(dataset):
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module( # resizing, mean과 std로 정규화하는 transform
                                resize=args.resize,
                                mean=dataset.mean,
                                std=dataset.std,
                                )
        

    return transform
def get_loss_optim(model):
    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()), #req_grad = True인 파라미터만 opt
        lr=args.lr,
        # weight_decay=5e-4
    )
    return criterion, optimizer
    
def get_logger(save_dir):
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:#./model/exp/config.json
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    return logger

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    
    """ np_images를 n개 표현하고 target label과 pred label 비교

    Args:
        np_images (np_array): 이미지
        gts (_type_): target label
        preds (_type_): pred_label
        n (int, optional): 표현할 이미지 개수
        shuffle (bool, optional): shuffle 여부

    Returns:
        _type_: plt.fig
    """
    batch_size = np_images.shape[0]
    assert n <= batch_size, "n is bigger than batch_size (default n = 16)"

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n)) # 0-batch_size에서 n개를 랜덤 선택, shuffle=False일 경우 0-n 
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T, fig size
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T, subplot 간격 조절
    n_grid = np.ceil(n ** 0.5) # 4
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices): # choice = index, shuffle or not
        gt = gts[choice].item() # label[choice]
        pred = preds[choice].item() # pred[choice]
        image = np_images[choice] # image[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt) # target_label -> mask_label, gender_label, age_label
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred) # pred_label -> mask_label, gender_label, age_label
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}" # mask - gt : target_mask_label, pred : pred_mask_label
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks) # target_mask_label, pred_mask_label, 'mask'
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title) # n=16 -> 4x4
        plt.xticks([]) # no xticks 
        plt.yticks([]) # no yticks 
        plt.grid(False) #격자 숨기기
        plt.imshow(image, cmap=plt.cm.binary)

    return figure # plt.figure 반환


def increment_path(path, exist_ok=False): # exp 폴더가 존재할시 expN 경로 리턴
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path) # ./model/exp
    if (path.exists() and exist_ok) or (not path.exists()): # path가 존재하고 exist_ok = True(덮어쓰기), or 존재하지 않음
        return str(path)
    else: # 중복을 허용하지 않을 시
        dirs = glob.glob(f"{path}*") # ./model/exp, ./model/exp1, ...
        matches = [re.search(r"%s(\d+)" % path.stem, d) for d in dirs] # exp1, exp2, ...
        i = [int(m.group(1)) for m in matches if m] # 1, 2, 3, ...
        n = max(i) + 1 if i else 1 # exp2부터 시작하던 걸 exp1부터 시작하도록 변경
        return f"{path}{n}"

class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def train(data_dir, model_dir, args):
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scaler = torch.cuda.amp.GradScaler()
    wandb.init(project="test-project", entity="boostcamp_nlp_10")
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name)) # ./model/exp

    model = get_model(device)
    dataset = get_dataset()
    transform = get_transform(dataset)
    dataset.set_transform(transform) # dataset에 transform 할당
    criterion, optimizer = get_loss_optim(model)
    scheduler = get_scheduler(optimizer)
    logger = get_logger(save_dir)

    best_val_acc = 0
    best_val_loss = np.inf # 무한
    best_val_f1 = 0

    early_stopping = EarlyStopping(patience = args.early_stop, verbose = True) # early stopping

    #train 
    train_set, val_set = dataset.split_dataset() # random split 

    # weight sampler
    # y_train_indices = train_set.indices
    # print(len(y_train_indices))
    # print(len(val_set.indices))

    # y_train = [dataset[i][1] for i in y_train_indices]

    # class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    # print(class_sample_count)
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[t] for t in y_train])
    # samples_weight = torch.from_numpy(samples_weight)
    # sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     sampler = sampler,
    #     num_workers=multiprocessing.cpu_count()//2, # cpu 절반 사용
    #     shuffle=False, #shuffle
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    train_loader = DataLoader( # random sampling
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2, # cpu 절반 사용
        shuffle=True, #shuffle
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True, # 왜 True로 되어 있지?
    )
    for epoch in range(args.epochs): # epoch 
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch # img, label
            inputs = inputs.to(device)
            
            labels = labels.to(device)

            optimizer.zero_grad()

            # using precision
            if args.precision=='True':
                with torch.cuda.amp.autocast():
                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)
            
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

            loss_value += loss.item() # loss 합
            matches += (preds == labels).sum().item() # 정답을 맞힌 수
            if (idx + 1) % args.log_interval == 0: # log_interval 마다 (default 20 step)
                train_loss = loss_value / args.log_interval # 20step loss의 평균
                train_acc = matches / args.batch_size / args.log_interval # 맞힌수 / batch_size / 20
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx) # tensorboard
                wandb.log({"Train/loss": train_loss, "Train/accuracy": train_acc})
                loss_value = 0
                matches = 0

        scheduler.step() # 매 epoch

        # val loop
        with torch.no_grad(): # 1 epoch train 끝나고
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_target = []
            val_labels = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item() # loss

                acc_item = (labels == preds).sum().item() # accuracy

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_target.extend(labels.tolist())
                val_labels.extend(preds.tolist())
                if figure is None:
                    # [1000, 3, 128, 96]
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    # [1000, 128, 96, 3]
                    inputs_np = dataset.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image( # inputs_np n개를 display하고 label 비교, profiledataset이면 non-shuffle
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    ) 

            val_loss = np.sum(val_loss_items) / len(val_loader) # 18900 * 0.2 // 1000
            val_acc = np.sum(val_acc_items) / len(val_set) # 3780
            val_f1 = f1_score(val_target, val_labels, average='macro')
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                best_val_acc = val_acc
            
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%} || "
                f"best acc : {best_val_acc:4.2%} || "
                f"[Val] f1 : {val_f1:4.2%} || "
                f"best f1 : {best_val_f1:4.2%} || "
                f"loss: {val_loss:4.2}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_figure("results", figure, epoch) # figure tensorboard에 저장
            wandb.log({"Val/loss": val_loss, "Val/accuracy":val_acc, "Val/f1": val_f1, "results": figure})
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping epoch : ", epoch)

                config_json = open(os.path.join(save_dir, 'config.json'), "r",encoding = 'utf')
                config = json.load(config_json)
                config_json.close()
                config["early stop epoch"] = epoch
                config["transform"] = str(transform.transform)

                config_json = open(os.path.join(save_dir, 'config.json'), "w",encoding = 'utf')
                json.dump(config, config_json, ensure_ascii=False, indent=4)
                config_json.close()
                break
            print() # ?


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=(128, 96), help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
    parser.add_argument('--model', type=str, default='densenet', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.00009, help='learning rate (default: 0.00009)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=1, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--weight_decay', type=float, default= 5e-4, help='optimizer weight decay(default: 5e-4)')
    parser.add_argument('--steplr_gamma', type=float, default= 0.1, help='StepLR gamma(default: 0.5)')

    parser.add_argument('--pretrained', type=str, default='True', help='use pretrained model (default : False)')
    parser.add_argument('--early_stop', type=int, default=1, help='early stop patience (default: 10)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # Bag of tricks args
    parser.add_argument('--LR_scheduler', type=str, default='StepLR', help='using cosine LR scheduler')
    parser.add_argument('--precision', type=str, default='True', help='using cosine FP16 precision')

    #albumentations 사용: pip install albumentations
    

    args = parser.parse_args()
    wandb.config = vars(args)

    # print(args)
    print(wandb.config)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
