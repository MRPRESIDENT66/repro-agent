# Code Critic RAG packet

Generated from the clean blind workspace.

## Query

openood networks resnet18_32x32.py ResNet18_32x32 class BasicBlock shortcut

## Results

Most relevant files:
  openood/networks/resnet18_32x32.py  —  class BasicBlock(nn.Module):
  configs/datasets/cifar10/cifar10_adversarial_ood.yml  —  ood_dataset:
  scripts/eval_ood.py  —  ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
  openood/networks/net_utils_.py  —  def get_network(network_config):
  openood/networks/__init__.py  —  try:

## Query

configs/datasets/cifar10/cifar10_ood.yml nearood test_cifar100 test_tin

## Results

Most relevant files:
  configs/datasets/cifar10/cifar10_ood.yml  —  ood_dataset:
  configs/datasets/cifar100/cifar100_ood.yml  —  ood_dataset:
  configs/datasets/cifar10/cifar10_fsood.yml  —  ood_dataset:
  configs/datasets/cifar100/cifar100_fsood.yml  —  ood_dataset:
  openood/evaluation_api/datasets.py  —  if tvs.__version__ >= '0.13':

## Query

configs/datasets/cifar10/cifar10.yml test_cifar10 normalization

## Results

Most relevant files:
  configs/datasets/cifar10/cifar10.yml  —  dataset:
  results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml  —  !!python/object/new:openood.utils.config.Config
  openood/evaluation_api/preprocessor.py  —  INTERPOLATION = tvs_trans.InterpolationMode.BILINEAR
  openood/preprocessors/pixmix_preprocessor.py  —  resize_list = {
  scripts/ad/patchcore/cifar10_test_ood_patchcore.sh  —  PYTHONPATH='.':$PYTHONPATH \

## Query

openood evaluators metrics AUROC roc_curve confidence sign

## Results

Most relevant files:
  openood/evaluators/metrics.py  —  def compute_all_metrics(conf, label, pred):
  openood/evaluators/ad_evaluator.py  —  class ADEvaluator():
  openood/evaluators/ood_evaluator.py  —  class OODEvaluator(BaseEvaluator):
  openood/evaluators/base_evaluator.py  —  def to_np(x):
  scripts/sweep/sweep_posthoc.sh  —  python ./scripts/sweep/sweep_posthoc.py \
