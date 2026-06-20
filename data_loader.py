import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# 自定义TSRD数据集类（适配文件名带标签的格式）
class TSRDDataset(Dataset):
    def __init__(self, img_dir, img_list, transform=None):
        self.img_dir = img_dir
        self.img_list = img_list
        self.transform = transform
        # 从文件名前3位提取标签
        self.labels = [int(img_name[:3]) for img_name in img_list]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_dataloader(batch_size=64):
    # 你的数据集路径
    train_root = r"E:\DATE\TSRD\TSRD-Train Annotation\tsrd-train"
    test_root = r"E:\DATE\TSRD\TSRD-Test Annotation\TSRD-Test"

    # 加载所有训练集图片
    all_train_imgs = [f for f in os.listdir(train_root) if f.endswith(('.jpg', '.png', '.jpeg'))]
    # 划分训练集:验证集 = 8:2
    train_imgs, val_imgs = train_test_split(all_train_imgs, test_size=0.2, random_state=2025)

    # 加载所有测试集图片
    test_imgs = [f for f in os.listdir(test_root) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 创建数据集
    train_dataset = TSRDDataset(train_root, train_imgs, train_transform)
    val_dataset = TSRDDataset(train_root, val_imgs, val_test_transform)
    test_dataset = TSRDDataset(test_root, test_imgs, val_test_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader