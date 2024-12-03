import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import gc


# 데이터셋 정의
class FocusBlurDataset(Dataset):
    def __init__(self, csv_file, transform=None, num_samples=None):
        data = pd.read_csv(csv_file, header=None)
        
        # 랜덤 샘플링
        if num_samples and num_samples < len(data):
            data = data.sample(n=num_samples, random_state=42).reset_index(drop=True)

        self.rgb_paths = data[0].tolist()
        self.depth_paths = data[1].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        img_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if self.transform:
            original_img = self.transform(image)
            depth_map = self.transform(depth_map)

        return original_img, depth_map


# UNet 블록 정의
def unet_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 업샘플 블록 정의
def upsample_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    )


# 모델 정의
class UNetModified(nn.Module):
    def __init__(self):
        super(UNetModified, self).__init__()
        
        # Downsample Path
        self.down1 = unet_block(3, 32, pool=True)  # 128x128 -> 64x64
        self.down2 = unet_block(32, 64, pool=True)  # 64x64 -> 32x32
        self.down3 = unet_block(64, 128, pool=True)  # 32x32 -> 16x16
        self.down4 = unet_block(128, 256, pool=True)  # 16x16 -> 8x8
        self.down5 = unet_block(256, 512, pool=True)  # 8x8 -> 4x4

        # Bottleneck
        self.bottleneck = unet_block(512, 1024, pool=False)  # 4x4

        # Upsample Path
        self.up1 = upsample_block(1024, 512)
        self.up_conv1 = unet_block(512 + 512, 512, pool=False)
        
        self.up2 = upsample_block(512, 256)
        self.up_conv2 = unet_block(256 + 256, 256, pool=False)
        
        self.up3 = upsample_block(256, 128)
        self.up_conv3 = unet_block(128 + 128, 128, pool=False)
        
        self.up4 = upsample_block(128, 64)
        self.up_conv4 = unet_block(64 + 64, 64, pool=False)
        
        self.up5 = upsample_block(64, 32)
        self.up_conv5 = unet_block(32 + 32, 32, pool=False)

        # 출력
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Downsample Path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Bottleneck
        bn = self.bottleneck(d5)

        # Upsample Path
        up1 = self.up1(bn)
        up1 = torch.cat((nn.functional.interpolate(up1, size=d5.shape[2:], mode='bilinear', align_corners=False), d5), dim=1)
        up1 = self.up_conv1(up1)

        up2 = self.up2(up1)
        up2 = torch.cat((nn.functional.interpolate(up2, size=d4.shape[2:], mode='bilinear', align_corners=False), d4), dim=1)
        up2 = self.up_conv2(up2)

        up3 = self.up3(up2)
        up3 = torch.cat((nn.functional.interpolate(up3, size=d3.shape[2:], mode='bilinear', align_corners=False), d3), dim=1)
        up3 = self.up_conv3(up3)

        up4 = self.up4(up3)
        up4 = torch.cat((nn.functional.interpolate(up4, size=d2.shape[2:], mode='bilinear', align_corners=False), d2), dim=1)
        up4 = self.up_conv4(up4)

        up5 = self.up5(up4)
        up5 = torch.cat((nn.functional.interpolate(up5, size=d1.shape[2:], mode='bilinear', align_corners=False), d1), dim=1)
        up5 = self.up_conv5(up5)

        # 출력
        output = self.final_conv(up5)
        return output


# 학습 함수
def train_model(csv_file, num_epochs=10, batch_size=4, lr=0.001, num_samples=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ])
    dataset = FocusBlurDataset(csv_file, transform=transform, num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetModified().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')  # 초기 Validation Loss를 무한대로 설정

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for original, depth in tqdm(train_loader, desc="Training"):
            original, depth = original.to(device), depth.to(device)

            optimizer.zero_grad()
            outputs = model(original)
            outputs = nn.functional.interpolate(outputs, size=depth.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, depth)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * original.size(0)

            # 메모리 정리
            del outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        train_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.10f}")

        # Validation
        model.eval()
        val_loss = 0.0
        for original, depth in tqdm(val_loader, desc="Validation"):
            original, depth = original.to(device), depth.to(device)
            with torch.no_grad():
                outputs = model(original)
                outputs = nn.functional.interpolate(outputs, size=depth.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, depth)
                val_loss += loss.item() * original.size(0)

            # 메모리 정리
            del outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.10f}")

        # Validation Loss가 줄어들면 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_modified_model.pth")
            print("Model saved as 'best_unet_modified_model.pth'")

    print("Training complete. Best Validation Loss:", best_val_loss)


if __name__ == '__main__':
    csv_file = 'data/nyu2_train.csv'
    train_model(csv_file, num_epochs=40, num_samples=10000)
