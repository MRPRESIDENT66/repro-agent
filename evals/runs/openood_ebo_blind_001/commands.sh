ls -la
ls -la results/
ls -la data/
find results/ -type f | head -30
find data/ -type f | head -40
cat results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml
find data/benchmark_imglist/cifar10/ -type f
find data/benchmark_imglist/ -type d | sort
cat data/benchmark_imglist/cifar10/test_cifar100.txt | head -5
cat data/benchmark_imglist/cifar10/test_tin.txt | head -5
cat data/benchmark_imglist/cifar10/test_cifar10.txt | head -5
find data/images_cl/ -type d | head -20
ls data/images_cl/cifar10/test/ | head -5
ls data/images/
find data/ -type d | sort
cat openood/pipelines/test_ood_pipeline.py
cat openood/evaluators/evaluator.py
find openood/evaluators/ -type f
cat openood/evaluators/ood_evaluator.py
cat openood/evaluators/metrics.py
find openood/postprocessors/ -type f
cat openood/postprocessors/ebo_postprocessor.py
find configs/ -type f -name "*.yml" | xargs grep -l "ebo\|EBO\|energy" 2>/dev/null | head -20
find configs/ -type f | head -40
cat configs/postprocessors/ebo.yml
cat configs/pipelines/test/test_ood.yml
find scripts/ -type f -name "*.sh" | head -20
cat scripts/ood/residual/cifar10_test_ood_residual.sh
find configs/datasets/cifar10/ -type f
cat configs/datasets/cifar10/cifar10.yml
cat configs/datasets/cifar10/cifar10_ood.yml
find configs/networks/ -type f | head -20
cat configs/networks/resnet18_32x32.yml
cat openood/pipelines/test_ood_pipeline.py
cat openood/evaluators/ood_evaluator.py
cat openood/datasets/__init__.py
cat openood/datasets/utils.py
cat openood/datasets/imglist_dataset.py
cat openood/preprocessors/test_preprocessor.py
cat openood/preprocessors/base_preprocessor.py
cat openood/preprocessors/transform.py
cat openood/pipelines/test_ood_pipeline.py
find . -name "main.py" -o -name "test_*.py" | head -10
cat openood/main.py
