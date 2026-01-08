import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from nets.resnet50_2_tinet import ResNet, Bottleneck
import argparse

parser = argparse.ArgumentParser(description='ResNet Test')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
parser.add_argument('--model', type=int, default=1, help='model number')
args = parser.parse_args()

# --- 하이퍼파라미터 및 경로 설정 (기존과 동일) ---
BATCH_SIZE = 64
NUM_WORKERS = 4
CUSTOM_CONV_LAYER_INDEX = args.cusin
WEIGHT_PATHS = [
    [
        "../weights/tinet/tinet_model-1_cusin-1_epoch-61.pth",
        "../weights/tinet/tinet_model-1_cusin-2_epoch-63.pth",
        "../weights/tinet/tinet_model-1_cusin-3_epoch-66.pth",
        "../weights/tinet/tinet_model-1_cusin-4_epoch-73.pth"
    ],
    [
        "../weights/tinet/tinet_model-2_cusin-1_epoch-56.pth",
        "../weights/tinet/tinet_model-2_cusin-2_epoch-57.pth",
        "../weights/tinet/tinet_model-2_cusin-3_epoch-37.pth",
        "../weights/tinet/tinet_model-2_cusin-4_epoch-59.pth"
    ]
]
WEIGHT_PATH = WEIGHT_PATHS[args.model - 1][CUSTOM_CONV_LAYER_INDEX - 1]

# --- Top-k 정확도 계산 함수 추가 ---
def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """상위 k개 예측 중 정답이 포함된 개수를 반환"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 상위 k개의 인덱스 추출
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # 정답 레이블과 비교 (정답을 예측 행렬 크기로 확장)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # k번째 행까지 중 하나라도 True가 있으면 정답으로 처리
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res # [Top-1 맞은 개수, Top-5 맞은 개수]

# --- 커스텀 데이터셋 클래스 (기존과 동일) ---
class TinyImageNetValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(root, 'images')
        self.annotations_file = os.path.join(root, 'val_annotations.txt')
        
        train_dir = os.path.join(os.path.dirname(root), 'train')
        if os.path.exists(train_dir):
            train_ds = datasets.ImageFolder(train_dir)
            self.class_to_idx = train_ds.class_to_idx
        else:
            print(f"Warning: Train directory not found. Mapping might be wrong.")
            self.class_to_idx = {} 

        self.data = []
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name, class_wnid = parts[0], parts[1]
                        if class_wnid in self.class_to_idx:
                            self.data.append((img_name, self.class_to_idx[class_wnid]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# --- 메인 실행부 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == 1: 
    model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
else:
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

if os.path.isfile(WEIGHT_PATH):
    model.load_state_dict(torch.load(WEIGHT_PATH, weights_only=True))
else:
    exit(f"No weight file found at '{WEIGHT_PATH}'")

# 데이터 로더 (검증 셋)
transform = transforms.ToTensor()
val_dataset = TinyImageNetValDataset(root="../tiny-imagenet-200/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 평가 루프
model.eval()
top1_correct = 0
top5_correct = 0
total = 0

print(f"Starting Evaluation on Validation Set ({len(val_dataset)} images)...")
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Top-1, Top-5 개수 계산
        t1, t5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
        
        top1_correct += t1
        top5_correct += t5
        total += labels.size(0)

# 최종 결과 출력
top1_acc = 100 * top1_correct / total
top5_acc = 100 * top5_correct / total

print(f"\n✨ Final Evaluation Results")
print(f"✅ Top-1 Accuracy: {top1_acc:.2f}%")
print(f"✅ Top-5 Accuracy: {top5_acc:.2f}%")