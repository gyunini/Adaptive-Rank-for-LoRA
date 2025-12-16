#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM, # Seq2Seq 모델 사용
    DataCollatorForSeq2Seq, # Seq2Seq용 데이터 콜레이터 사용
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)
from peft.tuners.adalora import AdaLoraLayer


# BART 모델에 맞는 TARGET_MODULES 설정 사용
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"
]


def parse_args():
    """스크립트 실행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        choices=["lora", "adalora"],
        required=True,
        help="어댑터 종류 (lora / adalora)",
    )
    parser.add_argument(
        "--budget",
        type=str,
        choices=["small", "large"],
        default="small",
        help="파라미터 예산: small≈0.32M, large≈1.27M. BART-large 기준 각각 0.08%(small), 0.32%(large) 설정에 따름.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_cnndailymail_custom",
        help="학습 결과 및 체크포인트 저장 경로",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )

    return parser.parse_args()


def get_peft_config(method: str, budget: str, total_step: int | None = None):
    """
    Seq2Seq 태스크에 맞는 LoRA 또는 AdaLoRA 설정을 반환합니다.
    (nlu-cnndailymail.py의 설정을 따릅니다.)
    """
    if method == "lora":
        if budget == "small":
            r = 1
            alpha = 32 # WebNLG 논문 설정 사용
        else:  # large
            r = 4
            alpha = 32 # WebNLG 논문 설정 사용

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, # Seq2Seq 태스크 타입
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.1, # WebNLG 논문 설정 사용
            bias="none",
            target_modules=TARGET_MODULES,
        )

    elif method == "adalora":
        # 논문 부록 E 의 설정을 반영 (Table 12)
        if budget == "small":
            init_r = 2
            target_r = 1
            final_rank = 72
        else:
            init_r = 6
            target_r = 4
            final_rank = 288

        if total_step is None:
            raise ValueError(
                "AdaLoRA requires `total_step` parameter. Please provide the total number of training steps."
            )

        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, # Seq2Seq 태스크 타입
            init_r=init_r,
            target_r=target_r,
            lora_alpha=32,    # WebNLG 논문 설정 사용
            lora_dropout=0.1,    # WebNLG 논문 설정 사용
            bias="none",
            target_modules=TARGET_MODULES,
            total_step=total_step,
            # ===== 논문 Table 8 의 SST-2 설정에서 가져온 하이퍼파라미터 (nlu-cnndailymail.py와 동일) =====
            tinit=5000,      # ti
            tfinal=85000,    # tf
            deltaT=100,      # ΔT
            beta1=0.85,      # 중요도 EMA
            beta2=0.85,
            orth_reg_weight=0.1,  # γ
            # =====================================
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return peft_config


def print_trainable_parameters(model):
    """모델의 학습 가능한 파라미터 수를 출력하고 비율을 계산합니다."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(
        f"Trainable params: {trainable} "
        f"({trainable/1e6:.2f}M)  |  "
        f"Total params: {total} "
        f"({total/1e6:.2f}M)  |  "
        f"Trainable%: {100 * trainable / total:.4f}"
    )

def map_name_to_layer_and_type(name: str):
    """
    BART 모델의 파라미터 이름에서 layer index와 논문 표기(W_q, W_k, ...)를 추출.
    Encoder와 Decoder 모두 처리 가능해야 합니다.
    """
    # 예시: model.encoder.layers.0.self_attn.q_proj.lora_A
    layer_match = re.search(r"\.(encoder|decoder)\.layers\.(\d+)\.", name)
    if not layer_match:
        return None, None
    
    # BART는 Encoder/Decoder를 구분하여 Layer Index를 매기므로, 구분을 위해 "E" 또는 "D"를 앞에 붙임
    arch_type = "E" if layer_match.group(1) == "encoder" else "D"
    layer_idx = int(layer_match.group(2))
    
    layer_key = f"{arch_type}{layer_idx}"

    module_type = None
    if "q_proj" in name:
        module_type = "$W_q$"
    elif "k_proj" in name:
        module_type = "$W_k$"
    elif "v_proj" in name:
        module_type = "$W_v$"
    elif "out_proj" in name:
        module_type = "$W_o$"
    elif "fc1" in name: # FFN 1번째 레이어
        module_type = "$W_{f1}$"
    elif "fc2" in name: # FFN 2번째 레이어
        module_type = "$W_{f2}$"

    return layer_key, module_type

def get_ranks_from_lora_E(model):
    """
    AdaLoRA의 lora_E 행렬에서 non-zero 개수를 세어 계층별 현재 랭크를 추정합니다.
    """
    ranks = {}

    for name, param in model.named_parameters():
        if "lora_E" not in name:
            continue

        layer_key, module_type = map_name_to_layer_and_type(name)
        if module_type is None:
            continue

        # non-zero 원소의 개수를 랭크로 사용 (AdaLoRA가 실제로 사용하는 방식)
        rank = int((param.abs() > 1e-6).sum().item())
        
        if layer_key not in ranks:
            ranks[layer_key] = {}
        ranks[layer_key][module_type] = rank

    return ranks


def main():
    args = parse_args()
    set_seed(args.seed)

    model_name = "facebook/bart-large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터셋 & 토크나이저
    print("1. 데이터셋 및 토크나이저 로드")
    raw_datasets = load_dataset("cnn_dailymail", "3.0.0")  # 요약 데이터셋
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    max_source_length = 512
    max_target_length = 128
    
    def preprocess_fn(examples):
        """데이터를 토크나이징하고 Seq2Seq 레이블을 준비합니다."""
        model_inputs = tokenizer(
            examples["article"],
            truncation=True,
            max_length=max_source_length,
        )
        labels = tokenizer(
            text_target=examples["highlights"],
            truncation=True,
            max_length=max_target_length,
        )

        label_ids = labels["input_ids"]
        # Seq2Seq 학습을 위해 패딩 토큰을 -100으로 대체
        label_ids = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    encoded = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=["article", "highlights", "id"],
    )
    
    # 2) 모델 로드
    print("2. 모델 로드")
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config
    )

    # 3) total_step 계산
    num_train_epochs = 15 # nlu-cnndailymail.py 설정
    per_device_train_batch_size = 32 # nlu-cnndailymail.py 설정
    train_dataset_size = len(encoded["train"])
    total_steps = (train_dataset_size // per_device_train_batch_size) * num_train_epochs

    print(f"train_dataset_size = {train_dataset_size}")
    print(f"total_steps (expected optimizer steps) = {total_steps}")

    # 4) LoRA / AdaLoRA 설정 및 모델 변환
    print("3. PEFT 설정 및 모델 변환")
    peft_config = get_peft_config(
        args.method,
        args.budget,
        total_step=total_steps if args.method == "adalora" else None,
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)

    print("==== Trainable parameter statistics ====")
    print_trainable_parameters(model)

    # 5) Data collator & metric & dataloader
    print("4. 데이터 콜레이터 및 평가지표 설정")
    # DataCollatorForSeq2Seq는 입력과 레이블을 패딩하고 디코더 입력을 준비합니다.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        padding="longest",
    )
    # 요약 태스크의 평가지표는 ROUGE
    metric = evaluate.load("rouge")

    train_dataloader = DataLoader(
        encoded["train"],
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        encoded["validation"],
        batch_size=32,
        shuffle=False,
        collate_fn=data_collator,
    )

    # 6) Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.01, # nlu-cnndailymail.py 설정 (WebNLG 0.01 사용)
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-4) # nlu-cnndailymail.py 설정

    warmup_steps = 500 # nlu-cnndailymail.py 설정
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    best_eval_rougeL = 0.0

    output_path = os.path.join(args.output_dir, f"{args.method}_{args.budget}")
    os.makedirs(output_path, exist_ok=True)
    
    # Generation arguments for evaluation
    generation_kwargs = {
        "max_length": max_target_length,
        "num_beams": 4, # 논문 설정
        "do_sample": False,
    }


    for epoch in range(num_train_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_train_epochs} =====")
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            global_step += 1

            # Seq2Seq 모델의 입력은 input_ids, attention_mask, labels, decoder_input_ids 등
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Label Smoothing을 Trainer 없이 구현하기 위해 Loss를 직접 계산해야 하지만,
            # 여기서는 편의상 모델이 반환하는 Loss를 사용합니다.
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            epoch_loss += loss.item()

            optimizer.step()

            if args.method == "adalora":
                # AdaLoRA의 랭크 업데이트 및 할당 로직 호출
                base = getattr(model, "base_model", None)
                if base is not None and hasattr(base, "update_and_allocate"):
                    base.update_and_allocate(global_step)

                    # ★★ progress print
                    if global_step % 1000 == 0: # 잦은 출력 방지를 위해 1000 step마다 출력
                        ranks = get_ranks_from_lora_E(model)
                        print(f"[AdaLoRA] Ranks at step {global_step}: {ranks}")

            lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(
                    f"Step {global_step}/{total_steps} - "
                    f"Avg. Epoch Loss: {avg_loss:.4f}"
                )


        # ====== Epoch 끝난 후 평가 ======
        model.eval()
        all_preds_ids = []
        all_labels_ids = [] # 최종적으로 모든 배치의 레이블을 담을 리스트
        eval_loss = 0.0

        print("\n--- Starting Evaluation ---")
        with torch.no_grad():
            for batch in eval_dataloader:
                # Loss 계산
                inputs_loss = {k: v.to(device) for k, v in batch.items()}
                loss_outputs = model(**inputs_loss)
                eval_loss += loss_outputs.loss.item()
                
                # Generation을 위한 입력 (labels 등은 제외)
                inputs_gen = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }

                # Generate summary
                generated_ids = model.generate(**inputs_gen, **generation_kwargs)
                
                all_preds_ids.extend(generated_ids.cpu().tolist())
                
                # --- [수정된 부분] 레이블 텐서의 길이를 max_target_length로 통일 ---
                current_labels_tensor = batch["labels"].cpu()
                current_batch_size, current_len = current_labels_tensor.shape
                
                # max_target_length (128)에 비해 현재 배치 길이가 짧을 경우 패딩이 필요함
                padding_needed = max_target_length - current_len

                if padding_needed > 0:
                    # 필요한 만큼 -100 (마스크 값)으로 패딩
                    padding_tensor = torch.full(
                        (current_batch_size, padding_needed), 
                        -100, 
                        dtype=current_labels_tensor.dtype
                    )
                    padded_labels = torch.cat([current_labels_tensor, padding_tensor], dim=1)
                elif padding_needed < 0:
                    # 현재 배치 길이가 max_target_length보다 긴 경우는 없어야 하지만, 있다면 자름
                    padded_labels = current_labels_tensor[:, :max_target_length]
                else:
                    padded_labels = current_labels_tensor
                    
                # 이제 padded_labels의 모든 행은 max_target_length와 동일한 길이를 가집니다.
                all_labels_ids.extend(padded_labels.tolist())
                # --- [수정 끝] ---
        
        # 2. Metric Computation (ROUGE)
        eval_loss = eval_loss / len(eval_dataloader)

        # all_labels_ids가 모두 동일한 길이를 가지므로, NumPy 배열로 변환 가능
        all_labels_array = np.array(all_labels_ids) 
        
        # -100을 pad_token_id로 치환 (디코딩 및 ROUGE 계산을 위해)
        all_labels_ids = np.where(
            all_labels_array != -100, 
            all_labels_array, 
            tokenizer.pad_token_id
        )
        
        decoded_labels = tokenizer.batch_decode(all_labels_ids.tolist(), skip_special_tokens=True)
        decoded_preds  = tokenizer.batch_decode(all_preds_ids, skip_special_tokens=True)
        
        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        # 결과를 백분율로 반환하고, rougeL을 주요 지표로 사용
        eval_metrics = {k: round(v * 100, 4) for k, v in result.items()}
        eval_rougeL = eval_metrics.get("rougeL", 0.0)
        
        print(f"Eval Loss: {eval_loss:.4f}")
        print(f"Eval ROUGE results: {eval_metrics}")


        # 3. 체크포인트 저장
        ckpt_dir = os.path.join(output_path, f"epoch-{epoch+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved checkpoint to {ckpt_dir}")

        # 4. Best Model 업데이트
        if eval_rougeL > best_eval_rougeL:
            best_eval_rougeL = eval_rougeL
            best_dir = os.path.join(output_path, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"New best ROUGE-L={best_eval_rougeL:.4f}, saved to {best_dir}")

    print("\n==== Training finished ====")
    print(f"Best eval ROUGE-L: {best_eval_rougeL:.4f}")


if __name__ == "__main__":
    main()