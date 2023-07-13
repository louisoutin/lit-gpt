import torch
from pydantic import BaseModel
from lightning.fabric.fabric import _PRECISION_INPUT


class FinetuneConfig(BaseModel):
    data_dir: str = "data/alpaca"
    checkpoint_dir: str = "checkpoints/stabilityai/stablelm-base-alpha-3b"
    out_dir: str = "out/lora/alpaca"

    precision: _PRECISION_INPUT | None = None
    tpu: bool = False

    eval_interval: int = 100
    save_interval: int = 100
    eval_iters: int = 100
    log_interval: int = 1
    devices: int = torch.cuda.device_count() or 1
    # change this value to force a maximum sequence length
    override_max_seq_length: int | None = None

    # Hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 128
    micro_batch_size: int = 4

    epoch_size: int = 50000  # train dataset size
    weight_decay: float = 0.01
    num_epochs: int = 5
    num_warmup_epochs: int = 2

    @property
    def gradient_accumulation_iters(self) -> int:
        accumulations = self.batch_size // self.micro_batch_size
        assert accumulations > 0
        return accumulations

    @property
    def max_iters(self) -> int:
        return (
            self.num_epochs * (self.epoch_size // self.micro_batch_size) // self.devices
        )

    @property
    def warmup_steps(self) -> int:
        return (
            self.num_warmup_epochs
            * (self.epoch_size // self.micro_batch_size)
            // self.devices
            // self.gradient_accumulation_iters
        )


class LoraFinetuneConfig(FinetuneConfig):
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
