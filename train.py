import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups: # param_group (dict) – Specifies what Tensors should be optimized along with group
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size # batch size가 n보다 작다면 AsserError를 발생시킴

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n)) # shuffle 할 것인가?
                                                                                    # Return a k sized list of population elements chosen with replacement
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5) # 정사각형 grid 형태를 짜기 위해서
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt) # label 값을 분해하는 함수. mask_label, gender_label, age_label 순서로 return 함.
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path) # PurePath subclass that can make system calls
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*") # Return a list of paths matching a pathname pattern
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

    # 나에게는 난잡한 메서드가 넘치는..increment_path() -> 나중에 다시 볼 것


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
                                                                      # 위 예시에서는 dataset 라이브러리의 MaskBaseDataset 클래스를 가져옴.
    dataset = dataset_module(
        data_dir=data_dir, 
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize, # default=[128, 96]
        mean=dataset.mean,  # defaults = MaskBaseDetaset의 mean=(0.548, 0.504, 0.479) 
        std=dataset.std, # default = MaskBaseDetaset의 std=(0.237, 0.247, 0.246)
    )
    dataset.set_transform(transform) # 클래스로 객체를 생성하여 set_transform 메서드를 실행함.

    # -- data_loader
    train_set, val_set = dataset.split_dataset()  # 위와 동일한 방식

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda, # use_cuda = torch.cuda.is_available()
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size, # valid와 train는 서로 다르게 batch size를 지정해주는 건가..?
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes # BaseModel의 __init__(self, num_classes)
    ).to(device) # 모델도 넣어줬으면 데이터도 넣어줘야지.. 근데 데이터는 어디에서 넣나..?
    model = torch.nn.DataParallel(model) # torch.nn.DataParallel(model, device_ids=[0, 1, 2]) 예시처럼, 나눠서 넣는 것도 가능함.

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
                                                                        # 아 이것을 통해서 내가 구현은 안했지만 기존의 구현된 라이브러리를 가져와서 연결할 수 있었음
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()), # filter(적용시킬 함수, 적용할 요소들) : 특정 조건으로 걸러진 요소들로 iterator객체로 만듦
                                                               # p.requires_grad가 True인 것들만 적용하는..
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5) # default=20

    # -- logging
    logger = SummaryWriter(log_dir=save_dir) # save_dir = increment_path(os.path.join(model_dir, args.name))
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device) # 위에서 모델은 따로 to(device)를 해줌
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer) # 위에서 정의한 함수
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad(): # 전체 데이터를 한 번 다 돌 때마다, valid로 검사해보는..
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader: # val_loader의 경우 train_loader와는 다르게 battch size가 다르던데..
                                         # 아무튼 default로 1000으로 되어있는데, 이게 맞나?
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                # 이미지를 복원해서 그림을 그려주는.
                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset" # 해당 클래스인 경우에는 섞지 않도록 설정함.
                    ) 

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc: # 점점 작은 loss 값이 나오는 순간을 저장함.
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth") # 작은 loss가 나오는 이전의 순간을 저장함
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch) # Add scalar data to summary
            logger.add_figure("results", figure, epoch) # Render matplotlib figure into an image and add it to summary
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv # Python-dotenv reads key-value pairs from a .env file and can set them as environment variables
    import os
    load_dotenv(verbose=True) # load_dotenv() : take environment variables from .env.

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args() # parser.parse_args()를 통해 parameter 변수에 접근하여 사용할 수 있음.
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)