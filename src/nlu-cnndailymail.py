import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch
from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoConfig,   #수정
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,  
    Seq2SeqTrainer,
)

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"
]

# 주석 추가
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
        help="파라미터 예산: small≈0.32M, large≈1.27M, BART-large 기준 각각 0.08%(small), 0.32%(large) 설정에 따름 ",  # Table 12 확인 시 참고
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_cnndailymail",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def get_peft_config(method: str, budget: str, total_step: int = None):
    if method == "lora":
        if budget == "small":
            r = 1 
            alpha = 32 # LoRA 논문에 cnn/dailymail 설정 언급이 없어서, NLG 태스크인 WebNLG 논문 설정(alpha = 32) 사용
        else:  # large
            r = 4
            alpha = 32   # LoRA 논문에 cnn/dailymail 설정 언급이 없어서, NLG 태스크인 WebNLG 논문 설정(alpha = 32) 사용

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.1, ## LoRA 논문에 cnn/dailymail 설정 언급이 없어서, NLG 태스크인 WebNLG 논문 설정(dropout prob = 0.01) 사용
            bias="none",
            target_modules=TARGET_MODULES,
        )

    elif method == "adalora":
        if budget == "small":
            init_r = 2
            target_r = 1
            final_rank = 72
        else:
            init_r = 6
            target_r = 4
            final_rank = 288

        if total_step is None:
            raise ValueError("AdaLoRA requires `total_step` parameter. Please provide the total number of training steps.")

        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            init_r=init_r,
            target_r=target_r,
            lora_alpha=32,    # LoRA 논문에 cnn/dailymail 설정 언급이 없어서, NLG 태스크인 WebNLG 논문 설정(alpha = 32) 사용
            lora_dropout=0.1,   # LoRA 논문에 cnn/dailymail 설정 언급이 없어서, NLG 태스크인 WebNLG 논문 설정(dropout prob = 0.01) 사용
            bias="none",
            target_modules=TARGET_MODULES,
            total_step=total_step,
            tinit=5000,     # ti
            tfinal=85000,   # tf
            deltaT=100,     # ΔT
            beta1=0.85,     # 중요도 EMA (언급은 없으나 NLU 실험 설정과 동일하게 0.85 사용)
            beta2=0.85,
            orth_reg_weight=0.1,  # γ 
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

    model_name = "facebook/bart-large"   #NLU 과제는 BART-large 사용

    raw_datasets = load_dataset("cnn_dailymail", "3.0.0")  # 3.0.0 버전 사용
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) 
    
    max_source_length = 512
    max_target_length = 128
    
    def preprocess_fn(examples):
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


    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config
    )

    # AdaLoRA를 위해 total_step 계산 필요
    num_train_epochs = 15
    per_device_train_batch_size = 32
    train_dataset_size = len(encoded["train"])
    # gradient_accumulation_steps는 기본값 1 사용
    total_steps = (train_dataset_size // per_device_train_batch_size) * num_train_epochs
    
    peft_config = get_peft_config(args.method, args.budget, total_step=total_steps if args.method == "adalora" else None)
    model = get_peft_model(model, peft_config)

    print("==== Trainable parameter statistics ====")
    print_trainable_parameters(model)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        # Seq2SeqTrainer는 preds가 tuple로 올 때가 있음 (sequences, ...)
        if isinstance(preds, tuple):
            preds = preds[0]

        # labels: -100 -> pad
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # ★ preds도 -100이나 음수 있으면 pad로 치환
        preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)

        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        return {k: round(v * 100, 4) for k, v in result.items()}


    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.method}_{args.budget}"),
        eval_strategy="epoch",   
        save_strategy="epoch",

        learning_rate=5e-4,                    # Table 8: 5 × 10^-4 : 5e-4
        per_device_train_batch_size=32,        # batch size 32
        per_device_eval_batch_size=32,

        num_train_epochs=15,                   # epochs 15
        weight_decay=0.01,                     # LoRA 논문에 cnn/dailymail 설정 언급이 없어서, NLG 태스크인 WebNLG 논문 설정(0.01) 사용
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        bf16=True,
        fp16=False,
        warmup_steps=500,                       #LoRA 논문에 cnn/dailymail 설정 언급이 없어서, NLG 태스크인 WebNLG 논문 설정(warmup steps = 500) 사용
        label_smoothing_factor=0.1,             #LoRA 논문 참고해서 Label Smoothing 적용

        predict_with_generate=True,
        generation_max_length=max_target_length,
        generation_num_beams=4,
        eval_accumulation_steps=4,
    )

    

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.model.config.max_length = max_target_length
    trainer.model.config.num_beams = 4

    trainer.train()
    eval_res = trainer.evaluate()
    print("==== Final eval on validation set ====")
    print(eval_res)


if __name__ == "__main__":
    main()
