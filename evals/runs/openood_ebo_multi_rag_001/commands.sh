ls scripts/eval_ood.py && head -80 scripts/eval_ood.py
ls scripts/ood/ebo/
ls configs/postprocessors/ | head -30
cat scripts/ood/ebo/cifar10_test_ood_ebo.sh
cat configs/postprocessors/ebo.yml
cat scripts/ood/ebo/cifar100_test_ood_ebo.sh
cat scripts/eval_ood.py
cat configs/pipelines/test/test_ood.yml
ls configs/datasets/cifar10/
cat configs/datasets/cifar10/cifar10.yml
cat configs/datasets/cifar10/cifar10_ood.yml
cat openood/evaluation_api/evaluator.py
cat openood/postprocessors/ebo_postprocessor.py
cat openood/evaluators/metrics.py
cat openood/pipelines/test_ood_pipeline.py
cat openood/evaluators/ood_evaluator.py
ls configs/networks/
cat configs/networks/resnet18_32x32.yml
find / -type d -name "results" 2>/dev/null | head -10
ls -la /workspace/OpenOOD/results/ 2>/dev/null || ls -la results/ 2>/dev/null || echo "checking other locations"
find /workspace/results -type f -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" 2>/dev/null
ls -la /workspace/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
cat scripts/ood/ebo/cifar10_test_ood_ebo.sh
cat scripts/ood/ebo/cifar100_test_ood_ebo.sh
cat scripts/eval_ood.py
cat openood/evaluation_api/evaluator.py
cat openood/evaluation_api/datasets.py
ls /workspace/data/ 2>/dev/null || ls data/ 2>/dev/null || echo "checking /workspace"
find /workspace -maxdepth 3 -type d -name "data" 2>/dev/null
ls /workspace/data/images_classic/ | head -20
ls /workspace/data/benchmark_imglist/cifar10/
cat scripts/ood/ebo/cifar10_test_ood_ebo.sh | grep -v '^#' | grep -v '^$'
cat scripts/ood/ebo/cifar100_test_ood_ebo.sh | grep -v '^#' | grep -v '^$'
cat navigator_report.md
ls -la
tail -100 navigator_transcript.jsonl
head -200 navigator_transcript.jsonl
find . -name "navigator_report.md" 2>/dev/null
cat scripts/ood/ebo/cifar10_test_ood_ebo.sh
cat configs/postprocessors/ebo.yml
cat scripts/eval_ood.py
cat openood/evaluation_api/evaluator.py
cat scripts/eval_ood.py
ls -la results/
ls -la results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
ls -la results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/
cat results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml
ls data/
ls data/benchmark_imglist/cifar10/
cat openood/pipelines/test_ad_pipeline.py
cat openood/evaluators/ood_evaluator.py
cat openood/evaluators/metrics.py
cat scripts/eval_ood.py | head -100
cat scripts/eval_ood.py | head -200 | tail -100
cat scripts/eval_ood.py | tail -50
cat scripts/eval_ood.py | head -60
cat openood/evaluation_api/evaluator.py
cat openood/evaluation_api/datasets.py
cat openood/evaluation_api/preprocessor.py
cat openood/postprocessors/ebo.py
find openood -name "*ebo*" -o -name "*EBO*"
cat openood/postprocessors/ebo_postprocessor.py
cat openood/pipelines/test_ood_pipeline.py
cat openood/evaluators/ood_evaluator.py
find configs -name "*cifar10*" -path "*ood*" -o -name "*cifar10*" -path "*ebo*"
cat configs/datasets/cifar10/cifar10_ood.yml
find configs -name "*resnet*" -path "*cifar10*"
find configs -name "*resnet*"
cat configs/networks/resnet18_32x32.yml
find configs/pipelines -name "*test*" -o -name "*ood*"
cat configs/pipelines/test/test_ood.yml
cat scripts/ood/ebo/cifar10_test_ood_ebo.sh
cat navigator_report.md
cat reproducer_public_log.txt
cat eval.py
ls -la
find . -name "eval.py" -o -name "eval_ood.py" 2>/dev/null
cat scripts/eval_ood.py
cat navigator_transcript.jsonl | python3 -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)[:3000]) for l in sys.stdin]" 2>/dev/null | head -100
cat openood/postprocessors/ebo_postprocessor.py
cat openood/evaluation_api/evaluator.py
cat openood/evaluation_api/postprocessor.py
cat openood/evaluators/metrics.py
cat scripts/eval_ood.py
cat openood/evaluation_api/evaluator.py
cat openood/evaluation_api/evaluator.py | head -400
cd openood/evaluation_api && grep -n "def eval_ood\|def eval_ood_fast\|def eval_ood_standard\|def _eval_ood\|def eval" evaluator.py
cd openood/evaluation_api && sed -n '251,400p' evaluator.py
cat scripts/eval_ood.py
cd openood/evaluation_api && sed -n '330,450p' evaluator.py
cat openood/evaluators/metrics.py
cat openood/postprocessors/ebo_postprocessor.py
cat openood/postprocessors/base_postprocessor.py
cat configs/postprocessors/ebo.yml
cat configs/datasets/ood/cifar100.yml 2>/dev/null || find configs -name "*.yml" | head -30
cat reproducer_public_log.txt
python3 -c "
import json
with open('navigator_transcript.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('role') == 'assistant':
            content = d.get('content', '')
            if 'REPRO_RESULT' in content or 'repro_result' in content.lower() or 'near_ood_auroc' in content:
                print(content[:2000])
                print('---')
"
