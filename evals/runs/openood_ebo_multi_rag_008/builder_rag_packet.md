# Reproducer Builder RAG packet

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

ImglistDataset constructor and CIFAR-10 test image list loading

## Results

Most relevant files:
  openood/datasets/imglist_dataset.py  —  ImageFile.LOAD_TRUNCATED_IMAGES = True
  configs/datasets/cifar10/cifar10.yml  —  dataset:
  openood/datasets/imglist_extradata_dataset.py  —  ImageFile.LOAD_TRUNCATED_IMAGES = True
  openood/datasets/imglist_augmix_dataset.py  —  ImageFile.LOAD_TRUNCATED_IMAGES = True
  openood/datasets/udg_dataset.py  —  ImageFile.LOAD_TRUNCATED_IMAGES = True

## Query

CIFAR-10 test preprocessing normalization resize center crop

## Results

Most relevant files:
  openood/preprocessors/test_preprocessor.py  —  class TestStandardPreProcessor(BasePreprocessor):
  configs/datasets/cifar10/cifar10.yml  —  dataset:
  openood/evaluation_api/preprocessor.py  —  INTERPOLATION = tvs_trans.InterpolationMode.BILINEAR
  openood/preprocessors/base_preprocessor.py  —  class BasePreprocessor():
  results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml  —  !!python/object/new:openood.utils.config.Config

## Query

CPU-safe EBO inference without eager optional postprocessor imports

## Results

Most relevant files:
  openood/postprocessors/base_postprocessor.py  —  class BasePostprocessor:
  configs/postprocessors/ebo.yml  —  postprocessor:
  openood/postprocessors/cfood_postprocessor.py  —  class CFOODPostprocessor(BasePostprocessor):
  scripts/ood/ebo/cifar10_test_ood_ebo.sh  —  PYTHONPATH='.':$PYTHONPATH \
  openood/evaluators/ood_evaluator.py  —  class OODEvaluator(BaseEvaluator):
