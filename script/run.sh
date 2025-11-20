#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodelist=ubuntu
#SBATCH --output=logs/lora_%j.out
#SBATCH --error=logs/lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

cd /home/leeg/Adaptive-Rank-for-LoRA

# UV 가상환경 활성화
source .venv/bin/activate

echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

CUDA_DEVICE=1 python src/glue-sst2.py --method lora --budget small
    
echo ""
echo "=== 테스트 완료 ==="
echo "완료: $(date)"
