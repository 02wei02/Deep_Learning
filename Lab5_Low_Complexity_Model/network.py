import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 使用GPU如果可用
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# class ImprovedCNN(nn.Module):
#     def __init__(self, num_classes=100):
#         super(ImprovedCNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, groups=16),  # 深度卷積
#             nn.Conv2d(32, 32, kernel_size=1),  # 點卷積
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=32),  # 深度卷積
#             nn.Conv2d(64, 64, kernel_size=1),  # 點卷積
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=64),  # 深度卷積
#             nn.Conv2d(128, 128, kernel_size=1),  # 點卷積
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
#         self.fc = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.global_pool(x)  # [N, 128, 1, 1]
#         x = x.view(x.size(0), -1)  # 展平
#         x = self.fc(x)
#         return x

# 数据增强
def get_transforms():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

#call model
### End

def load_model(MODEL_PATH):
    model = ImprovedCNN(num_classes=100).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # 加載 state_dict
    model.eval()  # 設定為評估模式
    return model