# Navigator RAG packet

Generated from the clean blind workspace.

## Query

official CIFAR-10 EBO OOD evaluation entry command and multiple checkpoint runs

## Results

Most relevant files:
  scripts/ood/ebo/cifar10_test_ood_ebo.sh  —  PYTHONPATH='.':$PYTHONPATH \
  openood/utils/config.py  —  def setup_config(config_process_order=('merge', 'parse_args', 'parse_refs')):
  scripts/ood/ebo/sweep_osr.py  —  config = [
  openood/pipelines/test_acc_pipeline.py  —  class TestAccPipeline:
  scripts/ood/ebo/cifar100_test_ood_ebo.sh  —  PYTHONPATH='.':$PYTHONPATH \

## Query

EBO energy score temperature logsumexp implementation

## Results

Most relevant files:
  openood/postprocessors/ebo_postprocessor.py  —  class EBOPostprocessor(BasePostprocessor):
  configs/postprocessors/ebo.yml  —  postprocessor:
  scripts/ood/ebo/cifar10_test_ood_ebo.sh  —  PYTHONPATH='.':$PYTHONPATH \
  scripts/ood/ebo/cifar100_test_ood_ebo.sh  —  PYTHONPATH='.':$PYTHONPATH \
  scripts/ood/ebo/imagenet200_test_ood_ebo.sh  —  python scripts/eval_ood.py \

## Query

CIFAR-10 near OOD CIFAR-100 TinyImageNet data loading and test preprocessing

## Results

Most relevant files:
  openood/evaluation_api/datasets.py  —  if tvs.__version__ >= '0.13':
  openood/evaluation_api/preprocessor.py  —  INTERPOLATION = tvs_trans.InterpolationMode.BILINEAR
  configs/datasets/cifar100/cifar100.yml  —  dataset:
  configs/datasets/cifar100/cifar100_adversarial_ood.yml  —  ood_dataset:
  scripts/ood/mcd/cifar100_test_mcd.sh  —  python main.py \

## Query

ResNet18 32x32 checkpoint loading and OOD AUROC metric aggregation

## Results

Most relevant files:
  scripts/sweep/sweep_posthoc.py  —  network_dict = {
  configs/datasets/cifar10/cifar10_adversarial_ood.yml  —  ood_dataset:
  openood/networks/utils.py  —  def get_network(network_config):
  openood/evaluators/metrics.py  —  def compute_all_metrics(conf, label, pred):
  scripts/eval_ood.py  —  ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

## Query

CPU compatibility CUDA calls in OpenOOD OOD inference

## Results

Most relevant files:
  openood/evaluators/ood_evaluator.py  —  class OODEvaluator(BaseEvaluator):
  openood/postprocessors/base_postprocessor.py  —  class BasePostprocessor:
  openood/pipelines/utils.py  —  def get_pipeline(config: Config):
  openood/networks/clip.py  —  def zeroshot_classifier(model, classnames, templates):
  openood/postprocessors/gradnorm_postprocessor.py  —  class GradNormPostprocessor(BasePostprocessor):
