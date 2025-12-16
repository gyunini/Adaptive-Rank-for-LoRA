import os
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from peft import PeftConfig
from transformers import AutoModelForSequenceClassification

# ==========================================
# 학습된 체크포인트 경로를 지정
# 예: ./outputs_sst2/adalora_small/checkpoint-50520
# CHECKPOINT_PATH = "./outputs_cola/adalora_small/epoch-25"
CHECKPOINT_PATH = "./outputs_cola/adalora_large/epoch-25"
# ==========================================


def map_name_to_layer_and_type(name: str):
    layer_match = re.search(r"layer\.(\d+)\.", name)
    if not layer_match:
        return None, None
    layer_idx = int(layer_match.group(1))

    module_type = None
    if "query_proj" in name:
        module_type = "$W_q$"
    elif "key_proj" in name:
        module_type = "$W_k$"
    elif "value_proj" in name:
        module_type = "$W_v$"
    elif "attention.output.dense" in name:
        module_type = "$W_o$"
    elif "intermediate.dense" in name:
        module_type = "$W_{f1}$"
    elif "output.dense" in name:
        module_type = "$W_{f2}$"

    return layer_idx, module_type


def get_ranks_from_rank_pattern(rank_pattern: dict):
    ranks = {}

    for name, mask in rank_pattern.items():
        layer_idx, module_type = map_name_to_layer_and_type(name)
        if module_type is None:
            continue

        # mask 타입별로 rank 계산
        if isinstance(mask, torch.Tensor):
            rank = int(mask.sum().item())
        else:
            # list[bool] or list[int]
            rank = int(sum(mask))

        if layer_idx not in ranks:
            ranks[layer_idx] = {}
        ranks[layer_idx][module_type] = rank

    return ranks


def get_ranks_from_lora_E(model):
    ranks = {}

    for name, param in model.named_parameters():
        if "lora_E" not in name:
            continue

        layer_idx, module_type = map_name_to_layer_and_type(name)
        if module_type is None:
            continue

        rank = int((param.abs() > 1e-6).sum().item())
        if layer_idx not in ranks:
            ranks[layer_idx] = {}
        ranks[layer_idx][module_type] = rank

    return ranks


def plot_heatmap(ranks_dict, output_filename="adalora_rank_heatmap.png"):
    df = pd.DataFrame(ranks_dict).sort_index(axis=1)  # layer 기준 정렬

    # 논문 Figure 3 순서
    row_order = ["$W_{f2}$", "$W_{f1}$", "$W_o$", "$W_v$", "$W_k$", "$W_q$"]
    row_order = [r for r in row_order if r in df.index]
    df = df.reindex(row_order)

    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.2)

    ax = sns.heatmap(df, annot=True, fmt="d", cmap="YlGn", cbar_kws={"label": "Final Rank"})
    plt.title("AdaLoRA Rank Allocation (DeBERTaV3-base)")
    plt.xlabel("Layer Index")
    plt.ylabel("Module Type")
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    print(f"Heatmap saved to {output_filename}")
    plt.show()


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint path not found: {CHECKPOINT_PATH}")
        return

    print(f"Loading PEFT config from {CHECKPOINT_PATH}...")
    peft_config = PeftConfig.from_pretrained(CHECKPOINT_PATH)
    print(f"peft_type      : {peft_config.peft_type}")
    print(f"base_model_name: {peft_config.base_model_name_or_path}")

    for attr in ["target_r", "init_r", "total_step", "tinit", "tfinal", "deltaT"]:
        if hasattr(peft_config, attr):
            print(f"{attr}: {getattr(peft_config, attr)}")

    rank_pattern = getattr(peft_config, "rank_pattern", None)
    # print("rank_pattern: >>>>>>>>>>>>>>>>>>>>>>> ", rank_pattern)
    if rank_pattern is not None:
        print(f"\n[rank_pattern detected] type={type(rank_pattern)}, entries={len(rank_pattern)}")
        first_key = next(iter(rank_pattern))
        first_val = rank_pattern[first_key]
        if isinstance(first_val, torch.Tensor):
            print(f"  example key: {first_key}, len={first_val.numel()}")
        else:
            print(f"  example key: {first_key}, len={len(first_val)}")
    else:
        print("\n[Warning] rank_pattern is None. "
              "update_and_allocate가 제대로 호출되지 않았거나, "
              "아직 rank allocation이 끝나기 전 checkpoint일 수 있습니다.")

    print("\nLoading full model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    if rank_pattern is not None:
        print("\nCalculating ranks from rank_pattern...")
        ranks = get_ranks_from_rank_pattern(rank_pattern)
    else:
        print("\nrank_pattern이 없어서 lora_E에서 직접 rank 추정...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        ranks = get_ranks_from_lora_E(model)

    if not ranks:
        print("No ranks found. Something is wrong (no AdaLoRA layers matched).")
        return

    print("\n[Layer-wise Rank Distribution]")
    df = pd.DataFrame(ranks).sort_index(axis=1).fillna(0).astype(int)
    print(df)

    # 시각화
    output_dir = os.path.dirname(CHECKPOINT_PATH)
    if output_dir == "":
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "adalora_rank_heatmap.png")
    plot_heatmap(ranks, output_path)


if __name__ == "__main__":
    main()
