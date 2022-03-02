import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

# albumentations
import albumentations
import albumentations.pytorch
import cv2

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename): #file이 IMG_EXTENSIONS으로 안끝나면 False
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([ # resize, tensor 변환, mean, std로 정규화
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image): # init은 생성할 때, call은 호출될 때 실행
        return self.transform(image) # a = BaseAugmentation(resize, mean, std, **) INIT
                                     # a(image) CALL

class AlbuAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = albumentations.Compose([
            albumentations.Resize(resize[0],resize[1]),
            albumentations.LongestMaxSize(max_size=max(resize)),
            albumentations.PadIfNeeded(min_height=max(resize),
                            min_width=max(resize),
                            border_mode=cv2.BORDER_CONSTANT),
            # albumentations.RandomCrop(width=resize[0], height=resize[1]),
            albumentations.OneOf([albumentations.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                                albumentations.VerticalFlip(p=1)            
                                ], p=1),
            albumentations.OneOf([albumentations.MotionBlur(p=1),
                                albumentations.OpticalDistortion(p=1),
                                albumentations.GaussNoise(p=1)                 
                                ], p=1),
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255),
            albumentations.pytorch.transforms.ToTensorV2()])

    def __call__(self, image): # init은 생성할 때, call은 호출될 때 실행
        image_transform = self.transform(image=image) # albumentation 결과는 dict형으로 반환됨
        return image_transform['image'].type(torch.float32) 


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor): # a(tensor) -> tensor에 noise를 추가해서 return
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self): # repr는 객체의 정보전달, str는 원하는 표현?
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            # CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            # ColorJitter(0.1, 0.1, 0.1, 0.1), # 밝기, 명도, 채도, 색조를 10%씩 +- 변화
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            # RandomGrayscale,
            # Grayscale(num_output_channels=3),
            Normalize(mean=mean, std=std),
            
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male": #male -> 0
            return cls.MALE
        elif value == "female": #femail -> 1
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        # elif value < 55:
        #     return cls.MIDDLE
        elif value < 60:
             return cls.MIDDLE
        else:
            return cls.OLD


class ModifyTrainData:
    duplicated_id = '003397-1'
    # delete_id = ['000225', '000357', '003798'] # 남녀 구분이 가지 않음
    mail_to_femail = ['001498-1','004432', '005223']
    femail_to_mail = ['006359','006360','006361','006362','006363','006364']
    # 기존 label 유지 ['000010','000667','000664','000725','000736','000767','000817','003780','004281','006504', '006424', '003223', '003113', '001509']
    # norm_to_incorrect = ['005227', '000020', '004418']
    age_to_young = ['001009', '001064', '001637', '001666', '001852', ]
    age_to_old = ['004348']
    @classmethod
    def modify_train_data(cls, train_df):
        train_df.loc[1367, 'id'] = '003397-1' # 중복 id

        for i in cls.mail_to_femail:
            train_df.loc[train_df['id'] == i,'gender'] = 'female' # mail -> femail

        for i in cls.femail_to_mail:
            train_df.loc[train_df['id'] == i,'gender'] = 'male' # femail -< mail

        for i in cls.age_to_old:
            train_df.loc[train_df['id'] == i,'age'] = 60 # middle -> old

        for i in cls.age_to_young:
            train_df.loc[train_df['id'] == i,'age'] = 29 # middle -> young

        # for i in cls.delete_id: # 남녀 구분 x
        #     del_idx = train_df[train_df['id'] == i].index
        #     train_df = train_df.drop(del_idx)
        
        # train_df.reset_index(drop = True, inplace=True) # 삭제된 data index를 비워놓기 때문에 index reset (0-2697)

        # for i in cls.norm_to_incorrect: # 파일명 확인
        #     fpath = str(train_df[train_df['id'] == i]['path'].values[0])

        #     path = os.path.join("/opt/ml/input/data/train/images/", fpath)
        #     incorrect_fname = os.path.join(path, "incorrect_mask.jpg")
        #     normal_fname = os.path.join(path, "normal.jpg")
        #     temp_fname = os.path.join(path, "temp.jpg")

        #     os.rename(incorrect_fname, temp_fname)
        #     os.rename(normal_fname, incorrect_fname)
        #     os.rename(temp_fname, normal_fname)
        return train_df
       

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, flag_strat, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, train_csv_path = '/opt/ml/input/data/train/train.csv'):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.train_df = pd.read_csv(train_csv_path)
        self.train_df = ModifyTrainData.modify_train_data(self.train_df) # 이상값 수정
        self.transform = None
        self.flag_strat= flag_strat
        if self.flag_strat == False: 
            self.setup()
        else :
            num_person = len(self.train_df) # 2700
            self.train_df['folder_class']=int(0)
            for i in range(num_person):
                gender = self.train_df.loc[i, 'gender']
                age = self.train_df.loc[i, 'age']

                gender_label = GenderLabels.from_str(gender) # GenderLabels.MALE (0 or 1)
                age_label = AgeLabels.from_number(age) # AgeLabels.YOUNG (0 or 1 or 2)
                kfold_class = self.encode_kfold_class(gender_label,age_label) # kfold_class 0: male/young ~ 5: female/old
                self.train_df.loc[i, 'folder_class']=kfold_class
        self.calc_statistics()

    def setup(self): # img_path, mask_label, gender_label, age_label
        """
        image_paths = full path (/opt/ml/input/data/train/images/000001_female_Asian_45/incorrect_mask.jpg)
        mask_labels = last path (incorrect_mask)
        gender_labels = df['gender']
        age_labels = df['age']
        """
        
        num_person = len(self.train_df)
        for i in range(num_person):

            gender = self.train_df.loc[i, 'gender']
            age = self.train_df.loc[i, 'age']

            gender_label = GenderLabels.from_str(gender) # GenderLabels.MALE (0 or 1)
            age_label = AgeLabels.from_number(age) # AgeLabels.YOUNG (0 or 1 or 2)
            label_path = self.train_df.loc[i, 'path']
            label_path = os.path.join(self.data_dir, label_path) # /opt/ml/input/data/train/images/000001_female_Asian_45

            picture_list = os.listdir(label_path) # incorrect_mask.jpg, ...
            picture_list = [fname for fname in picture_list if fname[0] != "."] # 임시파일 제외
            for j in picture_list:
                img_path = os.path.join(label_path, j) # full path
                # gender_label = GenderLabels.from_str(gender) # GenderLabels.MALE (0 or 1)
                # age_label = AgeLabels.from_number(age) # AgeLabels.YOUNG (0 or 1 or 2)
                mask_label = self._file_names[j.split('.')[0]]
                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None # 평균, 표준편차가 none이 아니면 True
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]: # 18900
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255 # 왜 255로 나누지?
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label) # 0~17로 변환
        
        image_transform = self.transform(image)
        # image_transform = image_transform.type(torch.uint8) # float을 uint8로 줄여 전송
        return image_transform, multi_class_label # input = index, output = img & label

    def __len__(self): # 18900
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index): 
        image_path = self.image_paths[index]
        if str(type(self.transform)) == "<class 'dataset.AlbuAugmentation'>": # albumentation는 numpy형 이미지를 이용
            transformed_image = cv2.imread(image_path)
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            return transformed_image
        
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label
    
    @staticmethod
    def encode_kfold_class(gender_label, age_label) -> int:
        return gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]: # decoding 
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def decode_kfold_class(kfold_label) -> Tuple[GenderLabels, AgeLabels]: # decoding 
        gender_label = (kfold_label // 3) % 2
        age_label = kfold_label % 3
        return gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0 # 255?
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8) # 0보다 작은 값 = 0, 255보다 큰 값 = 255, uint8 가능
        return img_cp # 카피 리턴

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio) # 18900 * 0.2 = 3780
        n_train = len(self) - n_val # 15120
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set #dataset 반환





class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, flag_strat, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list) # print(indices['any_key']) -> [], == {"train" = [], "val" = []}
        super().__init__(data_dir, flag_strat, mean, std, val_ratio)
        
    def setup(self, train_idx, valid_idx):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        split_profiles = {
            "train": self.train_idx,
            "val": self.valid_idx
        }
        
        cnt = 0
        for phase, indices in split_profiles.items(): # train, index_set
            for _idx in indices:
                profile = profiles[_idx] # 2700개 중 하나
                img_folder = os.path.join(self.data_dir, profile) # /opt/ml/input/data/train/images/000001_female_Asian_45
                for file_name in os.listdir(img_folder): 
                    _file_name, ext = os.path.splitext(file_name) # mask1.jpg
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (/opt/ml/input/data/train/images, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name] # MaskLabels.MASK

                    id, gender, race, age = profile.split("_") # csv 기반으로 변환
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

       
    def split_dataset(self) -> List[Subset]: # train = 2160, val = 540
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            # CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            # ColorJitter(0.1, 0.1, 0.1, 0.1), # 밝기, 명도, 채도, 색조를 10%씩 +- 변화
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            # RandomGrayscale,
            # Grayscale(num_output_channels=3),
            Normalize(mean=mean, std=std),
        ])
        # self.transform = CustomAugmentation

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]) 

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
