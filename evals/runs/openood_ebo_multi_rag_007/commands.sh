python eval_ebo.py --root results/cifar10_resnet18_32x32_base_e100_lr0.1_default --device cpu
cat navigator_report.md 2>/dev/null || echo "NOT FOUND"
cat reproducer_public_log.txt 2>/dev/null || echo "NOT FOUND"
cat eval_ebo.py 2>/dev/null || echo "NOT FOUND"
ls -la
python -c "import openood; print(openood.__version__)" 2>&1
ls results/ 2>/dev/null; echo "---"; ls data/ 2>/dev/null; echo "---"; pip show openood 2>&1 | head -5
ls results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
python -c "import openood; print(dir(openood))" 2>&1
ls results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/
ls results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/
ls results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/
find / -path "*/openood*" -name "*.py" 2>/dev/null | head -20
ls /workspace/openood/
ls /workspace/openood/postprocessors/
cat /workspace/openood/postprocessors/ebo_postprocessor.py
cat /workspace/openood/evaluators/__init__.py
cat /workspace/openood/evaluators/utils.py
cat /workspace/openood/evaluators/ood_evaluator.py
cat /workspace/openood/evaluators/metrics.py
cat eval_ebo.py
ls data/benchmark_imglist/ 2>/dev/null
ls data/benchmark_imglist/cifar10/
ls data/images_classic/
head -3 data/benchmark_imglist/cifar10/test_cifar10.txt
head -3 data/benchmark_imglist/cifar10/test_cifar100.txt
head -3 data/benchmark_imglist/cifar10/test_tin.txt
ls data/images_classic/cifar10/
ls data/images_classic/cifar100/
ls data/images_classic/tin/
ls data/images_classic/cifar10/test/ | head -5
ls data/images_classic/cifar100/test/ | head -5
ls /workspace/openood/networks/
