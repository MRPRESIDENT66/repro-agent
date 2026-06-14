import json
import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. Load model (requires import detectors side-effect for registration)
import detectors  # noqa: F401
import timm

device = torch.device("cpu")
model = timm.create_model("resnet18_cifar100", pretrained=True)
model.eval()
model.to(device)

# 2. Read preprocessing from model's pretrained_cfg
cfg = model.pretrained_cfg
mean = cfg["mean"]   # e.g. (0.4914, 0.4822, 0.4465)
std  = cfg["std"]    # e.g. (0.2023, 0.1994, 0.2010)
input_size = cfg["input_size"]  # e.g. (3, 32, 32)

transform = T.Compose([
    T.Resize(input_size[1:]),          # (32, 32)
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

# 3. Load CIFAR-100 test set (offline, from cache)
dataset = load_dataset(
    "uoft-cs/cifar100",
    split="test",
    cache_dir=None,          # use default cache
    trust_remote_code=True,
)

def preprocess(example):
    img = example["img"].convert("RGB")
    example["pixel_values"] = transform(img)
    return example

dataset = dataset.map(preprocess, remove_columns=["img", "coarse_label"])
dataset.set_format(type="torch", columns=["pixel_values", "fine_label"])

loader = DataLoader(dataset, batch_size=128, shuffle=False)

# 4. Evaluate
correct = 0
total = 0
with torch.no_grad():
    for batch in loader:
        images = batch["pixel_values"].to(device)
        labels = batch["fine_label"].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
result = {"metric": "top1_accuracy", "actual": accuracy, "num_examples": total}
print(f"REPRO_RESULT {json.dumps(result)}")
