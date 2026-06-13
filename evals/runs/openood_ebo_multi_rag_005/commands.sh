python eval_ebo.py
cat navigator_report.md 2>/dev/null || echo "NOT FOUND"
cat reproducer_public_log.txt 2>/dev/null || echo "NOT FOUND"
cat eval_ebo.py 2>/dev/null || echo "NOT FOUND"
cat navigator_report.md
ls -la /
find /workspace -maxdepth 4 -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" 2>/dev/null | head -30
ls /workspace/
find /workspace/results -type f -name "*.ckpt" -o -name "*.pth" -o -name "*.pt" 2>/dev/null
ls /workspace/data/ 2>/dev/null; ls /data/ 2>/dev/null; find / -maxdepth 3 -name "cifar*" -type d 2>/dev/null | head -10
cat /workspace/navigator_report.md
find /workspace/results -type f 2>/dev/null
ls /workspace/data/benchmark_imglist/ 2>/dev/null; ls /workspace/data/images_classic/ 2>/dev/null
ls /workspace/data/images_classic/cifar10/ 2>/dev/null | head -5
ls /workspace/data/images_classic/cifar100/ 2>/dev/null | head -5; ls /workspace/data/images_classic/tin/ 2>/dev/null | head -5
cat /workspace/eval_ebo.py
python3 -c "
import torch
ckpt = torch.load('/workspace/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt', map_location='cpu')
print(type(ckpt))
if isinstance(ckpt, dict):
    print(list(ckpt.keys()))
    for k in list(ckpt.keys())[:10]:
        v = ckpt[k]
        if hasattr(v, 'shape'):
            print(f'  {k}: {v.shape}')
        else:
            print(f'  {k}: {type(v)}')
"
python3 -c "
import torch
from torchvision.models.resnet import BasicBlock, ResNet

class ResNet18_32x32(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = torch.nn.Identity()

    def forward(self, x):
        return self.net(x)

model = ResNet18_32x32(num_classes=10)
ckpt = torch.load('/workspace/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt', map_location='cpu')
# Check if state_dict keys match
model_dict = model.state_dict()
model_keys = set(model_dict.keys())
ckpt_keys = set(ckpt.keys())
print('Missing in ckpt:', model_keys - ckpt_keys)
print('Extra in ckpt:', ckpt_keys - model_keys)
print('Match:', model_keys == ckpt_keys)
"
python3 -c "
import torch
from torchvision.models.resnet import BasicBlock, ResNet

# The checkpoint keys don't have 'net.' prefix, so the model is a plain ResNet
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
# Adjust first conv for 32x32
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

ckpt = torch.load('/workspace/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt', map_location='cpu')
model_dict = model.state_dict()
model_keys = set(model_dict.keys())
ckpt_keys = set(ckpt.keys())
print('Missing in ckpt:', model_keys - ckpt_keys)
print('Extra in ckpt:', ckpt_keys - model_keys)
print('Match:', model_keys == ckpt_keys)
"
python3 -c "
import torch
ckpt = torch.load('/workspace/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt', map_location='cpu')
# Check keys containing shortcut/downsample
for k in ckpt.keys():
    if 'shortcut' in k or 'downsample' in k:
        print(k)
"
python3 -c "import torch; print(torch.__version__); import torchvision; print(torchvision.__version__)"
python3 -c "
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

# Check what the current ResNet uses
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
print('layer2.0 keys:', [k for k in model.state_dict().keys() if 'layer2.0' in k][:10])
"
python3 -c "
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

# Let's look at the BasicBlock source to understand the naming
import inspect
print(inspect.getsource(BasicBlock._forward_impl))
"
python3 -c "
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
import inspect
print(inspect.getsource(BasicBlock.forward))
"
cat /workspace/navigator_report.md 2>/dev/null | head -200
cat /workspace/navigator_report.md 2>/dev/null | wc -l
cat /workspace/navigator_report.md 2>/dev/null | tail -n +100
cat /workspace/navigator_report.md
cat /workspace/eval_ebo.py
cd /workspace && python eval_ebo.py --root /workspace/results/cifar10_resnet18_32x32_base_e100_lr0.1_default --device cpu 2>&1
find /workspace/data -name "*.txt" -path "*imglist*" 2>/dev/null | head -20
ls /workspace/data/benchmark_imglist/cifar10/ 2>/dev/null; ls /workspace/data/benchmark_imglist/cifar100/ 2>/dev/null; ls /workspace/data/benchmark_imglist/osr_tin20/ 2>/dev/null
head -3 /workspace/data/benchmark_imglist/cifar10/test_cifar10.txt
