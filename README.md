# Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning

*final team project of optimization for AI 2025-2*

This repository contains the full training and evaluation pipeline for **Adaptive Rank for LoRA**.
The project investigates about the efficient way to optimize LoRA training with improved performance.

The repository consists of:
1. AdaLoRA paper code implementation
2. Suggested Modificated Method of Adaptive Rank Allocation : **Dynamic Rank Scheduling via Loss Plateau**
3. AdaLoRA + QLoRA Implementation
4. Codes for Analysis (rank_visualization, memory_calculation)
5. [Final Report](Optimization_For_AI_Final_Project_Report%20.pdf)

---
## Environment Setup (uv, slurm)

This project uses **uv and slurm environment** for fast and reproducible Python environment management.

### 1. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate
````

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

Python version is specified in `.pyproject --requires-python` and dependency versions are locked via `uv.lock`.

> ⚠️ CUDA-enabled PyTorch is required for training and inference.

---
## Repository Structure

```
.
├── script/                   # Example end-to-end execution scripts
├── src/
│   ├── custom/               # modified rank allocation method experiment : Dynamic Rank Scheduling via Loss Plateau
│   ├── src_qlora/            # adalora + qlora experiment implementation  (cola, mnli, qnli, sst2 included)
│   ├── glue-cola.py          # nlu task (cola) lora training loop
│   ├── glue-mnli.py          # nlu task (mnli) lora training loop
│   ├── glue_sst2.py          # nlu task (sst2) lora training loop
│   ├── nlu-cnndailymail.py   # nlg task (cnn/dailymail) lora training loop      
│   └── visualize_rank.py                    
├── analysis/                 # analysis for model rank visualization & memory calculation
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Experiment Overview

### 1. Baseline Experiment - AdaLoRA & LoRA comparison (`src/*.py`)

The baseline experiment compares **LoRA** and **AdaLoRA** without any quantization, to isolate the effect of adaptive rank allocation.  
All training and evaluation scripts are implemented in `src/*.py`, using a shared configuration and hyperparameters across tasks.  
We fine-tune the same backbone model on four GLUE datasets (MNLI, SST-2, CoLA, QNLI) and NLG task dataset (cnn/dailymail) report **accuracy, memory usage, and rank-allocation heatmaps** for each method.

Below is the table - comparing performance of LoRA and AdaLoRA
### Baseline Results: LoRA vs. AdaLoRA

| Task Type        | Task          | Metric  | Small Budget LoRA | Small Budget AdaLoRA | Large Budget LoRA | Large Budget AdaLoRA |
|------------------|---------------|---------|-------------------|----------------------|-------------------|----------------------|
| **NLU (GLUE)**   | MNLI          | Acc     | 0.9010            | **0.9049**           | 0.9045            | **0.9062**           |
|                  | SST-2         | Acc     | 0.9541            | **0.9599**           | 0.9576            | **0.9610**           |
|                  | CoLA          | Mcc     | 0.6871            | **0.6915**           | 0.6953            | **0.6992**           |
|                  | QNLI          | Acc     | 0.8856            | **0.9456**           | 0.9387            | **0.9460**           |
| **NLG (CNN/DM)** | cnn/dailymail | Rouge1  | 44.2432           | **44.3125**          | 44.3312           | **44.5662**          |
|                  |               | Rouge2  | 21.2376           | **21.3367**          | 21.3691           | **21.3989**          |
|                  |               | RougeL  | 30.7821           | **31.0128**          | 31.1024           | **31.0752**          |

and below is the example of visualization of AdaLoRA rank pruning (initial step -> final step)
<img width="1398" height="322" alt="image" src="https://github.com/user-attachments/assets/ee72f8a4-0c3b-44b6-9100-d49a5f8e8a49" />

---
### 2. Dynamic Rank Scheduling via Loss Plateau (`src/custom/`)

Optimizing rank allocation by starting training with a minimal rank and grow only when necessary by checking the loss plateau.

* Initialization: Start training with a minimal rank (r=2)
* Monitoring: Track Validation Loss during training
* Expansion: If loss plateaus (no improvement for N epochs)
  * Rank Expansion ( "r "←"r + " Δ"r" )
  * Allocate new parameters to escape the local minimum

### Result
* Baseline (LoRA, r=8): 0.69 - 0.70
* Adaptive LoRA (Ours, Max r=8): 0.6757

=> Performance is comparable but slightly lower (-2.5%) than the static baseline

---

## 3. Combining AdaLoRA with QLoRA Quantization (`src/src_qlora/`)

This stage, we experimented a new design of LoRA, combining AdaLoRA with QLoRA.
We aimed to apply quantization to AdaLoRA, which we expected to lead to the enhanced performance by concentrating capacity on the most important ranks.

<img width="500" height="360" alt="image" src="https://github.com/user-attachments/assets/915eda93-c209-40ea-9bd2-02b7810576ff" />

* Base PLM 4-bit Quantization (*QLoRA method)
* Adapter SVD / Dropout (*AdaLoRA Method) & Weight Update (*QLoRA Method)

### QLoRA + AdaLoRA Accuracy (GLUE NLU Tasks)

| Task Type       | Task  | Eval Metric | LoRA   | AdaLoRA  | QLoRA  | QLoRA + AdaLoRA |
|-----------------|-------|-------------|--------|----------|--------|-----------------|
| **NLU (GLUE)**  | MNLI  | acc         | 0.9010 | **0.9049** | 0.8977 | 0.9013          |
|                 | SST-2 | acc         | 0.9529 | 0.9599   | 0.9530 | **0.9644**      |
|                 | CoLA  | mcc         | 68.71  | 0.6915   | 0.6886 | **0.7027**      |
|                 | QNLI  | acc         | 0.8856 | 0.9455   | 0.9422 | **0.9464**      |


### QLoRA + AdaLoRA Memory Usage (GLUE NLU Tasks)

| Task Type      | Task  | Metric                   | LoRA                     | AdaLoRA                  | QLoRA                    | QLoRA + AdaLoRA          |
|----------------|-------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| **NLU (GLUE)** | MNLI  | base / adapter (MB)      | 351.76 MB / 0.64 MB      | 351.76 MB / 0.62 MB      | 67.55 MB / 0.64 MB       | **67.55 MB / 0.68 MB**   |
|                |       | trainable adapter param  | 334,083                  | 325,083                  | 333,314                  | 357,336                  |
|                | SST2  | base / adapter (MB)      | 351.76 MB / 0.64 MB      | 351.76 MB / 0.76 MB      | 67.55 MB / 0.64 MB       | **67.55 MB / 0.72 MB**   |
|                |       | trainable adapter param  | 333,314                  | 400,355                  | 333,314                  | 375,768                  |
|                | CoLA  | base / adapter (MB)      | 351.76 MB / 0.64 MB      | 351.76 MB / 0.77 MB      | 67.55 MB / 0.64 MB       | **67.55 MB / 0.78 MB**   |
|                |       | trainable adapter param  | 333,314                  | 404,954                  | 333,314                  | 410,328                  |
|                | QNLI  | base / adapter (MB)      | 351.76 MB / 0.64 MB      | 351.76 MB / 0.65 MB      | 67.55 MB / 0.64 MB       | **67.55 MB / 0.66 MB**   |
|                |       | trainable adapter param  | 333,314                  | 340,442                  | 333,314                  | 345,816                  |


---

## 4. Study Conclusion

There are two main conclusions. 

First, the combination of QLoRA and AdaLoRA achieves roughly a 5× reduction in memory usage, while maintaining accuracy that is comparable to, or on some tasks even better than, AdaLoRA. 
Second, we were able to confirm the potential that combining quantization with adaptive rank pruning can serve as a practical design option that recovers a substantial portion of the performance loss caused by quantization, while still preserving memory efficiency.



## Citation

During the whole cycle of our project, we referred to several papers.
Our citation list is below.

  
- **AdaLoRA (PEFT docs)**  
  Hugging Face PEFT Documentation – AdaLoRA  
  <https://huggingface.co/docs/peft/package_reference/adalora>

- **QLoRA (PEFT docs)**  
  Hugging Face PEFT Documentation – QLoRA  
  [<https://huggingface.co/docs/peft/package_reference/adalora>](https://huggingface.co/blog/4bit-transformers-bitsandbytes#making-llms-even-more-accessible-with-bitsandbytes-4-bit-quantization-and-qlora)

- **LoRA paper**  
  *LoRA: Low-Rank Adaptation of Large Language Models*  
  ```bibtex
  @article{hu2022lora,
    title={Lora: Low-rank adaptation of large language models.},
    author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu and others},
    journal={ICLR},
    volume={1},
    number={2},
    pages={3},
    year={2022}
  }
  ```

- **AdaLoRA paper**  
  *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*  
  
  ```bibtex
  @article{zhang2023adalora,
    title={Adalora: Adaptive budget allocation for parameter-efficient fine-tuning},
    author={Zhang, Qingru and Chen, Minshuo and Bukharin, Alexander and Karampatziakis, Nikos and He, Pengcheng and Cheng, Yu and Chen, Weizhu and Zhao, Tuo},
    journal={arXiv preprint arXiv:2303.10512},
    year={2023}
  }
  ```

- **QLoRA paper**  
  *QLoRA: Efficient Finetuning of Quantized LLMs*  
  ```bibtex
  @article{dettmers2023qlora,
    title={Qlora: Efficient finetuning of quantized llms},
    author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
    journal={Advances in neural information processing systems},
    volume={36},
    pages={10088--10115},
    year={2023}
  }
  ```

