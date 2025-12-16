# adaptive_lora.py
# -*- coding: utf-8 -*-

import math
from typing import Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

class AdaptiveLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        max_r: int = 8,
        r_init: int = 2,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        assert isinstance(base_linear, nn.Linear)
        assert 0 < r_init <= max_r

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.max_r = max_r
        self.r_eff = r_init        # 현재 활성화된 rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / float(r_init)

        # base weight는 freeze
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.max_r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.max_r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

    @torch.no_grad()
    def set_rank(self, new_r: int):
        new_r = max(1, min(new_r, self.max_r))
        if new_r == self.r_eff:
            return

        print(
            f"[AdaptiveLoRALinear] Rank changed: {self.r_eff} -> {new_r}. "
            f"Scaling fixed at {self.scaling:.4f}"
        )
        self.r_eff = new_r

    @torch.no_grad()
    def increase_rank(self, optimizer: Optional[optim.Optimizer] = None, delta_r: int = 2):
        old_r = self.r_eff
        new_r = min(self.r_eff + delta_r, self.max_r)

        if new_r <= old_r:
            return

        print(f"[AdaptiveLoRALinear] Increasing rank {old_r} -> {new_r}...")
        nn.init.kaiming_uniform_(self.lora_A[old_r:new_r, :], a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[:, old_r:new_r])

        # Optimizer State 초기화
        if optimizer is not None:
            # === Momentum은 0으로, Variance는 평균값으로 ===
            def smart_reset_optimizer_state(param, slice_dim, start_idx, end_idx):
                if param not in optimizer.state:
                    return
                
                state = optimizer.state[param]
                
                # Momentum 0.0으로 리셋
                if 'exp_avg' in state:
                    if slice_dim == 0:
                        state['exp_avg'][start_idx:end_idx, :] = 0.0
                    else:             
                        state['exp_avg'][:, start_idx:end_idx] = 0.0
                
                # Variance (exp_avg_sq) 기존 값의 평균으로 채움
                if 'exp_avg_sq' in state:
                    if slice_dim == 0:
                        existing_var = state['exp_avg_sq'][:start_idx, :].mean().item()
                        state['exp_avg_sq'][start_idx:end_idx, :] = existing_var
                    else:
                        existing_var = state['exp_avg_sq'][:, :start_idx].mean().item()
                        state['exp_avg_sq'][:, start_idx:end_idx] = existing_var
                    
                    print(f"  -> Grafted variance ({existing_var:.2e}) to new params.")

            smart_reset_optimizer_state(self.lora_A, 0, old_r, new_r)
            smart_reset_optimizer_state(self.lora_B, 1, old_r, new_r)

        self.r_eff = new_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)

        if self.r_eff <= 0:
            return result

        # 현재 r_eff 만큼만 잘라서 사용
        A_eff = self.lora_A[: self.r_eff, :]   
        B_eff = self.lora_B[:, : self.r_eff]

        x_dropped = self.lora_dropout(x)
        lora_out = (x_dropped @ A_eff.t()) @ B_eff.t()

        return result + self.scaling * lora_out


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    names = module_name.split(".")
    parent = root
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent, names[-1]


def apply_adaptive_lora(
    model: nn.Module,
    target_module_keywords: List[str],
    max_r: int = 8,
    r_init: int = 2,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
) -> List[AdaptiveLoRALinear]:
    adaptive_modules: List[AdaptiveLoRALinear] = []

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(key in module_name for key in target_module_keywords):
            continue

        parent, child_name = _get_parent_module(model, module_name)
        base_linear = getattr(parent, child_name)

        wrapped = AdaptiveLoRALinear(
            base_linear=base_linear,
            max_r=max_r,
            r_init=r_init,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        setattr(parent, child_name, wrapped)
        adaptive_modules.append(wrapped)
        print(f"[apply_adaptive_lora] Replaced {module_name} with AdaptiveLoRALinear")

    return adaptive_modules


class PlateauRankScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        modules: Iterable[AdaptiveLoRALinear],
        epsilon: float = 1e-3,
        patience: int = 1,
        rank_step: int = 2,
    ):
        self.optimizer = optimizer
        self.modules = list(modules)
        self.epsilon = epsilon
        self.patience = patience
        self.rank_step = rank_step
        
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        improved = self.best_loss - val_loss

        if improved > self.epsilon:
            self.best_loss = val_loss
            self.bad_epochs = 0
            print(
                f"[PlateauRankScheduler] Improved! best_loss={self.best_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )
            return False
        else:
            self.bad_epochs += 1
            print(
                f"[PlateauRankScheduler] No significant improvement (bad_epochs={self.bad_epochs}/{self.patience})"
            )

            if self.bad_epochs >= self.patience:
                increased = False
                
                for m in self.modules:
                    if m.r_eff < m.max_r:
                        before = m.r_eff
                        m.increase_rank(optimizer=self.optimizer, delta_r=self.rank_step)
                        after = m.r_eff
                        if after > before:
                            increased = True

                self.bad_epochs = 0
                if increased:
                    print(
                        f"[PlateauRankScheduler] >>> INCREASED RANK by {self.rank_step}. "
                        f"New ranks (sample): {self.modules[0].r_eff}"
                    )
                else:
                    print("[PlateauRankScheduler] All modules reached max rank.")
                
                return increased

            return False