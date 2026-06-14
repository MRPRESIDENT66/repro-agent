import detectors  # must come before timm — registers the model
import timm
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import json

# 1. Load model with trained weights
model = timm.create_model("resnet18_cifar100", pretrained=True)
model.eval()

# 2. Preprocessing from model's config (not assumed)
cfg = model.pretrained_cfg
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg['mean'], cfg['std']),
])

# 3. Dataset: correct fields and split
dataset = load_dataset("uoft-cs/cifar100", split="test")

def preprocess(example):
    example['img'] = transform(example['img'])
    return example

dataset = dataset.map(preprocess, remove_columns=['coarse_label'])
dataset.set_format(type='torch', columns=['img', 'fine_label'])

# 4. DataLoader
loader = DataLoader(dataset, batch_size=128)

# 5. Evaluate
correct = 0
total = 0
with torch.no_grad():
    for batch in loader:
        images, labels = batch['img'], batch['fine_label']
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
result = {"metric": "top1_accuracy", "actual": accuracy, "num_examples": total}
print(f"REPRO_RESULT {json.dumps(result)}")
