import os
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import lightning as L
import torch
from lightning.fabric.strategies import XLAStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.lora import mark_only_lora_as_trainable, lora_filter, GPT, Config
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    lazy_load,
    check_valid_checkpoint_dir,
    step_csv_logger,
    chunked_cross_entropy,
    get_new_version_path,
)
from lit_gpt.speed_monitor import SpeedMonitor, measure_flops, estimate_flops
from lit_gpt.finetune.config import LoraFinetuneConfig
from scripts.prepare_alpaca import generate_prompt
from torch.utils.tensorboard import SummaryWriter


def setup(
    config: LoraFinetuneConfig = LoraFinetuneConfig(),
):
    if config.precision is None:
        config.precision = "32-true" if config.tpu else "bf16-mixed"
    fabric_devices = config.devices
    if fabric_devices > 1:
        if config.tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            fabric_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            raise NotImplementedError
    else:
        strategy = "auto"

    logger = step_csv_logger(
        Path(config.out_dir).parent,
        Path(config.out_dir).name,
        flush_logs_every_n_steps=config.log_interval,
    )
    fabric = L.Fabric(
        devices=fabric_devices,
        strategy=strategy,
        precision=config.precision,
        loggers=logger,
    )
    fabric.print(f"Devices: {fabric_devices}")
    fabric.print(config)
    fabric.launch(main, config)


def main(fabric: L.Fabric, config: LoraFinetuneConfig):
    check_valid_checkpoint_dir(Path(config.checkpoint_dir))

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")
    # default `log_dir` is "runs" - we'll be more specific here

    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(config.out_dir, exist_ok=True)

    train_data = torch.load(Path(config.data_dir) / "train.pt")
    val_data = torch.load(Path(config.data_dir) / "test.pt")

    model_config = Config.from_name(
        name=Path(config.checkpoint_dir).name,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    checkpoint_path = Path(config.checkpoint_dir) / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = GPT(model_config)
        model.apply(model._init_weights)  # for the LoRA weights
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to LoRA weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    mark_only_lora_as_trainable(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    fabric.print(f"Number of trainable parameters: {num_params}")
    num_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    fabric.print(f"Number of non trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    model, optimizer = fabric.setup(model, optimizer)

    version = get_new_version_path(Path(config.out_dir))
    version_dir = Path(config.out_dir) / f"version_{version}"
    writer = SummaryWriter(str(version_dir / "tensorboard"))
    with open(str(version_dir / "hyperparameters.json"), "w") as json_file:
        json.dump(config.dict(), json_file, indent=4)
        json_file.close()

    train_time = time.time()
    train(
        fabric,
        config,
        model,
        optimizer,
        train_data,
        val_data,
        Path(config.checkpoint_dir),
        Path(config.out_dir),
        speed_monitor,
        writer,
    )
    fabric.print(f"Training time: {(time.time()-train_time):.2f}s")

    # Save the final LoRA checkpoint at the end of training
    save_path = Path(config.out_dir) / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    config: LoraFinetuneConfig,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
    writer: SummaryWriter,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(
        train_data, config.override_max_seq_length
    )

    validate(
        fabric, config, model, val_data, tokenizer, longest_seq_length, 0, writer
    )  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # estimated is too much of an optimistic estimate, left just for reference
        estimated_flops = estimate_flops(meta_model) * config.micro_batch_size
        fabric.print(
            f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}"
        )
        x = torch.randint(0, 1, (config.micro_batch_size, model.config.block_size))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(
            f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}"
        )
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.time()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    for iter_num in range(config.max_iters):
        if step_count <= config.warmup_steps:
            # linear warmup
            lr = config.learning_rate * step_count / config.warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            config,
            train_data,
            longest_seq_length,
            longest_seq_ix if iter_num == 0 else None,
        )

        is_accumulating = (iter_num + 1) % config.gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(
                input_ids, max_seq_length=max_seq_length, lm_head_chunk_size=128
            )
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / config.gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        t1 = time.time()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * config.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % config.log_interval == 0:
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )
            writer.add_scalar("training loss", loss.item(), iter_num)
            writer.add_scalar("training time ms", (t1 - iter_t0) * 1000, iter_num)

        if not is_accumulating and step_count % config.eval_interval == 0:
            t0 = time.time()
            val_loss = validate(
                fabric,
                config,
                model,
                val_data,
                tokenizer,
                longest_seq_length,
                iter_num,
                writer,
            )
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
            fabric.print(
                f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms"
            )
            fabric.barrier()
            writer.add_scalar("validation loss", val_loss, iter_num)
            writer.add_scalar("validation time ms", t1 * 1000, iter_num)
        if not is_accumulating and step_count % config.save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    config: LoraFinetuneConfig,
    model: GPT,
    val_data: List[Dict],
    tokenizer: Tokenizer,
    longest_seq_length: int,
    iter_num: int,
    writer: SummaryWriter,
) -> float:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(config.eval_iters)
    for k in range(config.eval_iters):
        input_ids, targets = get_batch(fabric, config, val_data, longest_seq_length)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce first 10 examples:

    for sample in val_data[: min(len(val_data), 10)]:
        fabric.print(f"Instruction: {sample['instruction']}")
        fabric.print(f"Input: {sample['input']}")
        fabric.print(f"Expected Output: {sample['output']}")
        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, device=model.device)
        max_returned_tokens = len(encoded) + 100
        output = generate(
            model,
            idx=encoded,
            max_returned_tokens=max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=0.8,
            eos_id=tokenizer.eos_id,
        )
        output = tokenizer.decode(output)
        fabric.print(f"LLM Answer: {output}")
        fabric.print("##End LLM Answer##")
        text_to_print: str = f"""Instruction:
{sample['instruction']}
Input:
{sample['input']}
Expected Output:
{sample['output']}
LLM Answer:
{output}"""
        writer.add_text("example prediction", text_to_print, iter_num)
        model.reset_cache()
    model.train()
    return val_loss.item()


def get_batch(
    fabric: L.Fabric,
    config: LoraFinetuneConfig,
    data: List[Dict],
    longest_seq_length: int,
    longest_seq_ix: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (config.micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # it's better to pad to a fixed seq length with XLA to avoid recompilation
    max_len = (
        max(len(s) for s in input_ids)
        if fabric.device.type != "xla"
        else longest_seq_length
    )

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_max_seq_length(
    data: List[Dict], override_max_seq_length: int | None = None
) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length
        if isinstance(override_max_seq_length, int)
        else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_lora_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    import pydantic_argparse

    parser = pydantic_argparse.ArgumentParser(
        model=LoraFinetuneConfig,
        prog="Lora Finetuning",
    )
    config = parser.parse_typed_args()
    setup(config=config)
