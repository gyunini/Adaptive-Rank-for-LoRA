import argparse
import os

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

try:
    from adaptive_lora import apply_adaptive_lora, PlateauRankScheduler
except ImportError:
    from custom_adaptive_rank import apply_adaptive_lora, PlateauRankScheduler


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
        "--output_dir",
        type=str,
        default="./outputs_cola_adaptive",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="학습 에폭 수",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--max_r",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--r_init",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--plateau_epsilon",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--plateau_patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--rank_step",
        type=int,
        default=2,
    )

    return parser.parse_args()


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


    raw_datasets = load_dataset("glue", "cola")
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
        remove_columns=["sentence", "idx"],
    )

    config = AutoConfig.from_pretrained(
        model_name,
        classifier_dropout=0.0,
        num_labels=2,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    adaptive_modules = apply_adaptive_lora(
        model,
        target_module_keywords=TARGET_MODULES,
        max_r=args.max_r,
        r_init=args.r_init,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Freeze & Unfreeze
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("classifier.") or name.startswith("pooler."):
            param.requires_grad = True

    for m in adaptive_modules:
        m.lora_A.requires_grad = True
        m.lora_B.requires_grad = True

    model.to(device)

    print("\nInitializing Optimizer...")
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Rank Scheduler
    rank_scheduler = PlateauRankScheduler(
        optimizer=optimizer,
        modules=adaptive_modules,
        epsilon=args.plateau_epsilon,
        patience=args.plateau_patience,
        rank_step=args.rank_step,
    )

    print("==== Trainable parameter statistics ====")
    print_trainable_parameters(model)
    print(f"Initial ranks (sample): {adaptive_modules[0].r_eff}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("glue", "cola")

    train_dataloader = DataLoader(
        encoded["train"],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        encoded["validation"],
        batch_size=64,
        shuffle=False,
        collate_fn=data_collator,
    )

    num_train_epochs = args.num_train_epochs
    train_dataset_size = len(encoded["train"])
    steps_per_epoch = train_dataset_size // args.per_device_train_batch_size
    total_steps = steps_per_epoch * num_train_epochs

    print(f"train_dataset_size = {train_dataset_size}")
    print(f"steps_per_epoch = {steps_per_epoch}")
    print(f"total_steps = {total_steps}")

    warmup_steps = int(total_steps * 0.1)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    best_eval_mcc = -1.0 

    os.makedirs(args.output_dir, exist_ok=True)

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
            lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(
                    f"Step {global_step}/{total_steps} - "
                    f"epoch_loss_avg: {avg_loss:.4f}"
                )

        # ====== Evaluation ======
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
        current_mcc = eval_metrics['matthews_correlation']
        
        eval_loss = eval_loss / len(eval_dataloader)

        print(f"Eval loss: {eval_loss:.4f}  |  Eval MCC: {current_mcc:.4f}")
        print(f"Current ranks (sample): {adaptive_modules[0].r_eff}")

        rank_increased = rank_scheduler.step(eval_loss)

        if rank_increased:
             print(f"*** Rank Increased! Optimizer states reset. ***")

        ckpt_dir = os.path.join(
            args.output_dir,
            f"epoch-{epoch+1}",
        )
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if current_mcc > best_eval_mcc:
            best_eval_mcc = current_mcc
            best_dir = os.path.join(args.output_dir, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"New best MCC={best_eval_mcc:.4f}, saved to {best_dir}")

    print("\n==== Training finished ====")
    print(f"Best eval MCC: {best_eval_mcc:.4f}")


if __name__ == "__main__":
    main()