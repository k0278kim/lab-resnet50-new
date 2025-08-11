# train_imagenet_amp.py
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.resnet50_2_imagenet import ResNet2_imagenet, Bottleneck2_imagenet
from nets.early_stopping import EarlyStopping

# ----------------------------
# 설정 (환경에 맞게 조절)
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 32           # GPU 메모리에 따라 조절 (현재 64가 가능한 상태라면 OK)
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
NUM_WORKERS = 2            # 공유메모리 이슈 있으면 0 또는 1로 낮춤
PIN_MEMORY = False         # shm 이슈 있으면 False (속도는 약간 느려짐)
ACCUM_STEPS = 1            # gradient accumulation steps (메모리가 부족하면 >1로 설정)
CUSTOM_CONV_LAYER_INDEX = 1
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------
# cudnn 튜닝
# ----------------------------
torch.backends.cudnn.benchmark = True  # 입력 크기가 고정이면 성능 도움
# torch.backends.cudnn.deterministic = True  # 재현성 필요하면 True (속도 저하 가능)

# ----------------------------
# 모델 / 옵티마이저 / 스케일러
# ----------------------------
model = ResNet2_imagenet(
    Bottleneck2_imagenet,
    [3, 4, 6, 3],
    num_classes=1000,
    custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# ----------------------------
# 데이터셋 / DataLoader
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 일반적으로 ImageNet 표준 정규화 권장:
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

train_dataset = datasets.ImageFolder("/data/imagenet/train", transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)

# ----------------------------
# 조기종료(옵션)
# ----------------------------
early_stopping = EarlyStopping(patience=5, delta=0.001)

# ----------------------------
# 학습 루프 (AMP + optional accumulation)
# ----------------------------
try:
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        total_samples = 0
        epoch_start = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        optimizer.zero_grad()

        for step, (images, labels) in pbar:
            # 비동기 전송 (pin_memory=True일 때만 효과적)
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # AMP 자동 혼합 정밀도
            with autocast():
                outputs = model(images)
                loss_raw = criterion(outputs, labels)          # 실제 loss
                loss = loss_raw / ACCUM_STEPS                  # accumulation 고려

            # scaled backward
            scaler.scale(loss).backward()

            # 통계
            running_loss += loss_raw.item() * images.size(0)  # raw loss 기준으로 누적
            total_samples += images.size(0)

            # gradient accumulation step
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                # optimizer step via scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # tqdm 정보
            avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
            pbar.set_postfix({'batch_loss': f"{loss_raw.item():.4f}", 'avg_loss': f"{avg_loss:.4f}"})

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        print(f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s - Avg Loss: {epoch_avg_loss:.4f}")

        # 체크포인트 저장 (모델 + 옵티마이저 상태)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"resnet-imagenet-cidx{CUSTOM_CONV_LAYER_INDEX}-epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': epoch_avg_loss
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # 조기 종료 체크
        early_stopping(epoch_avg_loss)
        if early_stopping.early_stop:
            print(f"⛔ Early stopping at epoch {epoch+1}")
            break

except KeyboardInterrupt:
    print("Interrupted by user — saving current model...")
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(CHECKPOINT_DIR, "interrupt_checkpoint.pth"))
    print("Saved interrupt checkpoint. Exiting.")
