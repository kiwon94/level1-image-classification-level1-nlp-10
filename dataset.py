import os
import random
from collections import defaultdict
from enum import Enum # enum 클래스를 사용하면 인스턴스의 종류를 제한할 수 있음
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

# 파일이 이미지 확장자를 가지는 지 확인하는 함수, 아무대도 안 쓰임.. 개별적으로 쓰려고 만든 듯.
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR), # PIL Image or a torch Tensor 형태의 이미지를 resize 함
            ToTensor(),                     # Convert a PIL Image or numpy.ndarray to tensor.
                                            # 굳이 ToTensor 먼저 안해도 되나 보네..
            Normalize(mean=mean, std=std),
        ])

    # BaseAugmentation의 객체를 ( ) 연산자와 호출하게 되면 return을 반환함.
    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean # 표준정규분포에서 추출한 값에 표준편차 곱하고 평균을 더함.

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# 위에 BaseAugmentation이 있는데 따로 커스텀해서 만들어줌.
class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),                 # PIL Image or a torch Tensor 형태의 이미지 중앙을 해당 사이즈로 crop함 
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),        # 밝기, 대비, 채도, 색상 값으로 랜덤하게 바꿔주는?
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


# 마스크 라벨링
class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2

# 성별 라벨링
class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")

# 나이 라벨링
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
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD

# 기본 데이터셋 클래스
# 필수로 구성해야 하는 것 : __init__, __getitem__, __len__
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

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        # mean과 std를 다음과 같이 설정한 이유는? ->  ImageNet data의 mean과 std을 default 값으로 설정함.
        # val_ratio는 train 데이터의 일부를 valid 데이터로 나누기 위한 비율을 나타냄.
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None # 아래에 set_transform이 있는데 왜 None으로 두었을까?
        self.setup()
        self.calc_statistics()

    # 기본적인 데이터 셋업
    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                # 위에서 classmethod로 설정했기 때문에 아래와 같이 진행함. 클래스명 자체가 cls가 되는..
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                # 위에 빈 리스트를 설정함.
                self.image_paths.append(img_path) # 이미지 경로, 단순 문자열
                self.mask_labels.append(mask_label) # 마스크 라벨링 클래스를 통해 얻은 값을 추가함.
                self.gender_labels.append(gender_label) # 성별 라벨링 클래스를 통해 얻은 값을 추가함.
                self.age_labels.append(age_label) # 나이 라벨링 클래스르 통해 얻은 값을 추가함.

    # 통계량을 계산하는데 어디에 쓰이는 걸까? -> normalized 할 때?
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]: # 왜 3000개 단위로 끊었을까?

                # from PIL import Image 를 활용하여 이미지를 열고 numpy로 바꿈. type도 int32로 변경함.
                image = np.array(Image.open(image_path)).astype(np.int32) 

                # 가로와 세로로 평균을 구하는 데 왜 구하는 걸까?
                sums.append(image.mean(axis=(0, 1))) # E(x)
                squared.append((image ** 2).mean(axis=(0, 1))) # E(x^2)

            self.mean = np.mean(sums, axis=0) / 255 # 표본평균의 평균
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255 # 표본평균의 표준편차
                                                                                # Var(x) = E(x^2) - E(x)^2
                                                                                # sigam(x) = Var(x)^0.5

    def set_transform(self, transform):
        self.transform = transform # 아마도 인자값으로 함수 객체가 들어와야 할 듯

    def __getitem__(self, index):
        # assert : 조건이 True가 아니면 AssertError를 발생함.
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        # 아래 함수가 정의되어 있음.
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label) # 함수 객체에 실행 연산자 ( )를 적용하여서 값이 반환됨.

        image_transform = self.transform(image) # 위에서 self.transform = None이라고 있는데 아마도 None 부분이 함수 객체로 들어오기에 이와 같이 작성됨.
        return image_transform, multi_class_label 

    def __len__(self):
        return len(self.image_paths)

    # '->'의 의미 : return값이 어떠한 상태인지를 아려줌.
    # 아래의 4개 함수 모두 index로 접근하여 데이터를 가져옴 -> 이게 나중에 __getitem__에서 한 번에 데이터셋의 하나의 정보를 가져오게 함.
    # 모두 리스트에서 index로 접근함.
    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index] 

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    # 이것만 단순 문자열로 리스트에 값을 집어놨기에 -> 형태가 없음.
    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path) # from PIL import Image 를 활용하여 이미지를 열고

    # 정적 메소드는 항상 데코레이터로 만들어짐.
    # 이것을 하게 되면 클래스(또는 객체)에서 해당 메서드로 접근할 때, 메서드가 실행되는 것이 아닌 함수 객체를 반환함. 
    # 실행을 하려면 실행 연산자 ( )로 접근해야 함.
    # 이것을 왜 하나? -> 메모리 낭비를 방지하기 위함?
    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label
    

    # from typing import Tuple, List를 통해 type annotation을 추가할 수 있음.
    # 파이썬 내장 자료 구조에 대한 타입을 명시할 수 있음.
    # label 값을 분해하는 함수
    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3    # 0, 1, 2
        gender_label = (multi_class_label // 3) % 2  # 0, 1
        age_label = multi_class_label % 3            # 0, 1, 2
        return mask_label, gender_label, age_label

    # 이미지 복원
    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0 # 255를 제일 마지막에 나눠주었늗네... 이것을 먼저 곱해주지는 않네? 왜 그럴까?
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8) # 0보다 작은 값은 모두 0으로 만들고,
                                                          # 255보다 큰 값은 모두 255로 만듦
        return img_cp

    # 데이터 셋이다 보니깐 나눠주는 함수를 정의함.
    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio) # len(self.image_paths) * self.val_ratio
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val]) # from torch.utils.data import Dataset, Subset, random_split
        return train_set, val_set

# MaskBaseDataset을 상속함. 
class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio) 

        val_indices = set(random.choices(range(length), k=n_val)) # 전체 사람을 기준으로 k개의 인덱스를 선택함.
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")] 
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items(): # train_indices와 val_indices가 리스트 형태로 들어가 있음
                                                      # for문이 2번 돌아감. phase가 처음에는 "train", 그 다음은 "val"
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder): # 14장의 이미지가 존재.
                    _file_name, ext = os.path.splitext(file_name) # 확장자명을 분리
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name] # 위에 class의 속성으로 존재함. 딕셔너리 형태로 구현되어 있음

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt) # ??
                    cnt += 1

    def split_dataset(self) -> List[Subset]: # from torch.utils.data import Dataset, Subset, random_split
        return [Subset(self, indices) for phase, indices in self.indices.items()] # Subset을 이용하면, 기존 set이 업데이트 되면 subset도 자동 업데이트 됨.


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR), # 픽셀들 간의 선형 보간법.. 약간 덜 선명해지는
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
