#!/bin/bash
#SBATCH --job-name=test_lora
#SBATCH --nodelist=ubuntu
#SBATCH --output=logs/lora_%j.out
#SBATCH --error=logs/lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# cd /home/ahnjm/adalora/Adaptive-Rank-for-LoRA
cd /home/leeg/Adaptive-Rank-for-LoRA

# UV 가상환경 활성화
source .venv/bin/activate

echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

CUDA_DEVICE=2 python src/glue-mnli.py --method adalora --budget small
    
echo ""
echo "=== 테스트 완료 ==="
echo "완료: $(date)"
