import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

densenet = models.densenet201
resnet50 = models.resnet50
resnet152 = models.resnet152
wide_resnet101 = models.wide_resnet101_2
vgg16 = models.vgg16

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)



# Resnet
class Resnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet101", pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

class Resnet200(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet200", pretrained=True)
        self.model.classifier = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)



# EfficienNetModel
class Model_Efficientnet_b3a(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = timm.create_model("efficientnet_b3a", pretrained=True)
        self.model.classifier = nn.Linear(in_features=1536, out_features=num_classes, bias=True)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)

class Model_Efficientnet_b1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = timm.create_model("efficientnet_b1", pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes, bias=True)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)


# from coatnet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4
# class CoatNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.model = coatnet_0()
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)
#     def forward(self, x):
#         return self.model(x)

        
# if __name__ == '__main__':
#    model = Model_Efficientnet_b1(18)
#    print(model)

