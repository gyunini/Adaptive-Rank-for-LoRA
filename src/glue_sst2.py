import argparse
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)


TARGET_MODULES = [
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


def get_peft_config(method: str, budget: str, total_step: int | None = None):
    if method == "lora":
        if budget == "small":
            r = 2
            alpha = 8
        else:
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
        if budget == "small":
            init_r = 12
            target_r = 2
            final_rank = 144
        else:
            init_r = 12
            target_r = 8
            final_rank = 576

        if total_step is None:
            raise ValueError(
                "AdaLoRA requires `total_step` parameter. Please provide the total number of training steps."
            )

        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            init_r=init_r,
            target_r=target_r,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGET_MODULES,
            total_step=total_step,
            tinit=6000,
            tfinal=22000,
            deltaT=100,
            beta1=0.85,
            beta2=0.85,
            orth_reg_weight=0.1,
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
    set_seed(args.seed)

    model_name = "microsoft/deberta-v3-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    config = AutoConfig.from_pretrained(
        model_name,
        classifier_dropout=0.0,
        num_labels=2,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    num_train_epochs = 16 # 24
    per_device_train_batch_size = 8 # 32
    train_dataset_size = len(encoded["train"])
    total_steps = (train_dataset_size // per_device_train_batch_size) * num_train_epochs

    print(f"train_dataset_size = {train_dataset_size}")
    print(f"total_steps (expected optimizer steps) = {total_steps}")

    peft_config = get_peft_config(
        args.method,
        args.budget,
        total_step=total_steps if args.method == "adalora" else None,
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)

    print("==== Trainable parameter statistics ====")
    print_trainable_parameters(model)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("glue", "sst2")

    train_dataloader = DataLoader(
        encoded["train"],
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        encoded["validation"],
        batch_size=64,
        shuffle=False,
        collate_fn=data_collator,
    )

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.01,
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=6e-5) # 8e-4

    warmup_steps = int(total_steps * 0.1)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    best_eval_acc = 0.0

    os.makedirs(os.path.join(args.output_dir, f"{args.method}_{args.budget}"), exist_ok=True)

    for epoch in range(num_train_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_train_epochs} =====")
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            global_step += 1

            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k not in ["labels", "label", "idx"]}

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            epoch_loss += loss.item()

            optimizer.step()

            if args.method == "adalora":
                base = getattr(model, "base_model", None)
                if base is not None and hasattr(base, "update_and_allocate"):
                    base.update_and_allocate(global_step)

            lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(
                    f"Step {global_step}/{total_steps} - "
                    f"epoch_loss_avg: {avg_loss:.4f}"
                )


        # ====== Epoch 끝난 후 평가 ======
        model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0.0

        with torch.no_grad():
            for batch in eval_dataloader:
                labels = batch["labels"]
                inputs = {k: v for k, v in batch.items() if k not in ["labels", "label", "idx"]}

                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs, labels=labels)
                logits = outputs.logits
                loss = outputs.loss

                eval_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())


        eval_metrics = metric.compute(predictions=all_preds, references=all_labels)
        eval_loss = eval_loss / len(eval_dataloader)

        print(f"Eval loss: {eval_loss:.4f}  |  Eval accuracy: {eval_metrics['accuracy']:.4f}")

        ckpt_dir = os.path.join(
            args.output_dir,
            f"{args.method}_{args.budget}",
            f"epoch-{epoch+1}",
        )
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved checkpoint to {ckpt_dir}")

        if eval_metrics["accuracy"] > best_eval_acc:
            best_eval_acc = eval_metrics["accuracy"]
            best_dir = os.path.join(
                args.output_dir, f"{args.method}_{args.budget}", "best"
            )
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"New best acc={best_eval_acc:.4f}, saved to {best_dir}")

    print("\n==== Training finished ====")
    print(f"Best eval accuracy: {best_eval_acc:.4f}")


if __name__ == "__main__":
    main()
