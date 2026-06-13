# Reviewer RAG packet

Generated from the clean blind workspace.

## Query

audit EBO logsumexp score and AUROC sign convention

## Results

Most relevant files:
  openood/evaluators/metrics.py  —  def compute_all_metrics(conf, label, pred):
  openood/postprocessors/ebo_postprocessor.py  —  class EBOPostprocessor(BasePostprocessor):
  scripts/ood/ebo/cifar10_test_ood_ebo.sh  —  PYTHONPATH='.':$PYTHONPATH \
  openood/evaluation_api/evaluator.py  —  class Evaluator:
  openood/recorders/cutpaste_recorder.py  —  class CutpasteRecorder:

## Query

audit CIFAR-10 test preprocessing and near OOD benchmark image lists

## Results

Most relevant files:
  openood/evaluation_api/preprocessor.py  —  INTERPOLATION = tvs_trans.InterpolationMode.BILINEAR
  openood/evaluation_api/datasets.py  —  if tvs.__version__ >= '0.13':
  data/benchmark_imglist/cifar10/fractals_fvis.txt  —  fractals_and_fvis/first_layers_resized256_onevis/images/5167.png
  configs/datasets/cifar10/cifar10.yml  —  dataset:
  openood/preprocessors/test_preprocessor.py  —  class TestStandardPreProcessor(BasePreprocessor):
