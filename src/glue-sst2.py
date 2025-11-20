#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)


TARGET_MODULES = [
    # DeBERTaV3-base에서 논문처럼 Wq, Wk, Wv, Wo, Wf1, Wf2 에 해당하는 모듈 이름들
    "query_proj",
    "key_proj",
    "value_proj",
    "o_proj",
    "intermediate.dense",
    "output.dense",
]


def parse_args():
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
        help="파라미터 예산: small≈0.32M, large≈1.27M",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_sst2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def get_peft_config(method: str, budget: str, total_step: int = None):
    """
    budget:
      - small: 논문 0.32M 수준 (LoRAr=2, b(T)=144)
      - large: 논문 1.27M 수준 (LoRAr=8, b(T)=576)
    """
    if method == "lora":
        if budget == "small":
            r = 2
            alpha = 8
        else:  # large
            r = 8
            alpha = 32

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGET_MODULES,
        )

    elif method == "adalora":
        # 논문 부록 C 의 설정을 반영
        # b(T)=144 -> 평균 rank 2, b(T)=576 -> 평균 rank 8
        # 초기 rank r는 12로 두고 target_r를 2/8 로 맞춰서
        # "초기 넉넉한 rank -> 점진적 pruning" 구조를 흉내낸다.
        if budget == "small":
            init_r = 12
            target_r = 2
            final_rank = 144
        else:
            init_r = 12
            target_r = 8
            final_rank = 576

        if total_step is None:
            raise ValueError("AdaLoRA requires `total_step` parameter. Please provide the total number of training steps.")

        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            init_r=init_r,
            target_r=target_r,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGET_MODULES,
            total_step=total_step,
            # ===== 논문 Table 8 의 SST-2 설정 =====
            tinit=6000,     # ti
            tfinal=22000,   # tf
            deltaT=100,     # ΔT
            beta1=0.85,     # 중요도 EMA
            beta2=0.85,
            orth_reg_weight=0.1,  # γ (orthogonality regularizer)
            # =====================================
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return peft_config


def print_trainable_parameters(model):
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


def main():
    args = parse_args()

    model_name = "microsoft/deberta-v3-base"

    # 1) 데이터셋 & 토크나이저
    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def preprocess_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=128,
        )

    encoded = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=["sentence"],
    )

    # 2) 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # 3) LoRA / AdaLoRA 설정
    # AdaLoRA를 위해 total_step 계산 필요
    num_train_epochs = 24
    per_device_train_batch_size = 32
    train_dataset_size = len(encoded["train"])
    # gradient_accumulation_steps는 기본값 1 사용
    total_steps = (train_dataset_size // per_device_train_batch_size) * num_train_epochs
    
    peft_config = get_peft_config(args.method, args.budget, total_step=total_steps if args.method == "adalora" else None)
    model = get_peft_model(model, peft_config)

    print("==== Trainable parameter statistics ====")
    print_trainable_parameters(model)

    # 4) Data collator & metric
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("glue", "sst2")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        result = metric.compute(predictions=preds, references=labels)
        return result

    # 5) TrainingArguments — 논문 Table 8 (SST-2) 반영
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.method}_{args.budget}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=8e-4,                # Table 8: 8 × 10^-4 : 8e-4
        per_device_train_batch_size=32,    # batch size 32
        per_device_eval_batch_size=64,
        num_train_epochs=24,               # #epochs 24
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) Train & evaluate
    trainer.train()
    eval_res = trainer.evaluate()
    print("==== Final eval on validation set ====")
    print(eval_res)


if __name__ == "__main__":
    main()
