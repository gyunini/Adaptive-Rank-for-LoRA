#!/bin/bash
#SBATCH --job-name=test_adalora_cola
#SBATCH --nodelist=devbox
#SBATCH --output=logs/lora_%j.out
#SBATCH --error=logs/lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=100:00:00

cd /home/ahnjm/aioptim_adalora/Adaptive-Rank-for-LoRA

# UV 가상환경 활성화
source .venv/bin/activate

echo "memory test -- outputs_qnli/adalora_small/best"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

CUDA_DEVICE=1 python model_memory.py
    
echo ""
echo "=== 테스트 완료 ==="
echo "완료: $(date)"
