from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import DatasetRef, SFTDatasetManifest, SFTExample, SFTTrainingConfig, TrajectoryDatasetManifest, TrajectoryEpisode


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sft_dataset(dataset_dir: Path) -> tuple[SFTDatasetManifest, list[SFTExample]]:
    manifest_path = dataset_dir / "sft_dataset_manifest.json"
    manifest = SFTDatasetManifest.model_validate(_load_json(manifest_path))
    examples = [
        SFTExample.model_validate(_load_json(dataset_dir / str(relative_ref)))
        for relative_ref in manifest.example_refs
    ]
    return manifest, examples


def load_trajectory_dataset(dataset_dir: Path) -> tuple[TrajectoryDatasetManifest, list[TrajectoryEpisode]]:
    manifest_path = dataset_dir / "trajectory_dataset_manifest.json"
    manifest = TrajectoryDatasetManifest.model_validate(_load_json(manifest_path))
    episodes = [
        TrajectoryEpisode.model_validate(_load_json(dataset_dir / str(relative_ref)))
        for relative_ref in manifest.episode_refs
    ]
    return manifest, episodes


def load_training_config(path: Path) -> SFTTrainingConfig:
    return SFTTrainingConfig.model_validate(_load_json(path))


def _resolve_dataset_ref(root: Path, ref: DatasetRef) -> Path:
    path = Path(ref.path)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def validate_sft_training_config(root: Path, config: SFTTrainingConfig) -> dict[str, object]:
    missing_required_datasets: list[str] = []
    dataset_summaries: list[dict[str, object]] = []
    for ref in [*config.dataset_refs, *config.eval_dataset_refs]:
        resolved = _resolve_dataset_ref(root, ref)
        exists = resolved.exists()
        if ref.required and not exists:
            missing_required_datasets.append(ref.path)
            continue
        if not exists:
            dataset_summaries.append({"path": ref.path, "exists": False, "dataset_kind": ref.dataset_kind})
            continue
        if ref.dataset_kind == "sft":
            manifest, examples = load_sft_dataset(resolved)
            dataset_summaries.append(
                {
                    "path": ref.path,
                    "exists": True,
                    "dataset_kind": ref.dataset_kind,
                    "example_count": manifest.example_count,
                    "task_count": len(manifest.task_ids),
                    "split": manifest.split,
                }
            )
        elif ref.dataset_kind == "trajectory":
            manifest, episodes = load_trajectory_dataset(resolved)
            dataset_summaries.append(
                {
                    "path": ref.path,
                    "exists": True,
                    "dataset_kind": ref.dataset_kind,
                    "episode_count": manifest.episode_count,
                    "task_count": len(manifest.task_ids),
                    "split": manifest.split,
                }
            )
        else:
            dataset_summaries.append({"path": ref.path, "exists": True, "dataset_kind": ref.dataset_kind})
    return {
        "status": "ok" if not missing_required_datasets else "failed",
        "config_id": config.config_id,
        "model_id": config.model_id,
        "adapter_mode": config.adapter_mode,
        "missing_required_datasets": missing_required_datasets,
        "datasets": dataset_summaries,
    }


def _sft_refs_for_split(config: SFTTrainingConfig, split: str) -> list[DatasetRef]:
    return [ref for ref in config.dataset_refs if ref.dataset_kind == "sft" and (ref.split or "").lower() == split.lower()]


def _format_example_for_sft(example: SFTExample) -> str:
    return "\n".join(
        [
            "### Prompt",
            example.prompt.strip(),
            "",
            "### Response",
            example.response.strip(),
        ]
    )


def build_sft_training_rows(root: Path, config: SFTTrainingConfig) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {"train": [], "dev": []}
    for split in ("train", "dev"):
        for ref in _sft_refs_for_split(config, split):
            dataset_dir = _resolve_dataset_ref(root, ref)
            if not dataset_dir.exists():
                continue
            _, examples = load_sft_dataset(dataset_dir)
            for example in examples:
                rows[split].append(
                    {
                        "text": _format_example_for_sft(example),
                        "task_id": example.task_id,
                        "prompt_family": example.prompt_family,
                        "training_example_kind": example.metadata.get("training_example_kind"),
                        "source_episode_ref": example.source_episode_ref,
                    }
                )
    return rows


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()[:80]


def _emit_training_stage(stage: str, **fields: object) -> None:
    payload = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "stage": stage,
    }
    payload.update(fields)
    print(json.dumps(payload, default=str), flush=True)


def _build_peft_config(adapter_mode: str):
    from peft import LoraConfig, TaskType

    rank = 16 if adapter_mode == "qlora" else 8
    alpha = 32 if adapter_mode == "qlora" else 16
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )


def _build_model_load_kwargs(adapter_mode: str, *, use_cuda: bool):
    import torch
    from transformers import BitsAndBytesConfig

    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_cuda:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
        total_gib = max(1, int(torch.cuda.get_device_properties(0).total_memory / (1024**3)))
        kwargs["max_memory"] = {
            0: f"{max(1, total_gib - 4)}GiB",
            "cpu": "24GiB",
        }
        kwargs["offload_state_dict"] = True
    if adapter_mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )
    return kwargs


def run_sft_training_job(
    root: Path,
    config_path: Path,
    *,
    model_override: str | None = None,
    max_steps_override: int | None = None,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config = load_training_config(config_path)
    rows = build_sft_training_rows(root, config)
    out_dir = root / config.output_dir
    if model_override:
        out_dir = out_dir / f"override_{_safe_slug(model_override)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "training_run_report.json"
    effective_model_id = model_override or config.model_id
    effective_tokenizer_id = model_override or config.tokenizer_id or effective_model_id
    train_rows = rows["train"]
    dev_rows = rows["dev"]

    payload: dict[str, object] = {
        "config_id": config.config_id,
        "model_id": config.model_id,
        "effective_model_id": effective_model_id,
        "effective_tokenizer_id": effective_tokenizer_id,
        "adapter_mode": config.adapter_mode,
        "started_at": datetime.now(tz=UTC).isoformat(),
        "train_example_count": len(train_rows),
        "dev_example_count": len(dev_rows),
        "status": "blocked",
    }

    try:
        _emit_training_stage(
            "tokenizer_load_begin",
            effective_model_id=effective_model_id,
            effective_tokenizer_id=effective_tokenizer_id,
            adapter_mode=config.adapter_mode,
        )
        tokenizer = AutoTokenizer.from_pretrained(effective_tokenizer_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        _emit_training_stage("tokenizer_load_done", effective_tokenizer_id=effective_tokenizer_id)

        use_cuda = torch.cuda.is_available()
        model_kwargs = _build_model_load_kwargs(config.adapter_mode, use_cuda=use_cuda)
        if use_cuda:
            offload_dir = out_dir / "hf_offload"
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_dir)

        _emit_training_stage("model_load_begin", effective_model_id=effective_model_id, use_cuda=use_cuda, model_kwargs=model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(effective_model_id, **model_kwargs)
        _emit_training_stage("model_load_done", effective_model_id=effective_model_id)
        if config.adapter_mode == "qlora":
            _emit_training_stage("prepare_kbit_begin")
            model = prepare_model_for_kbit_training(model)
            _emit_training_stage("prepare_kbit_done")

        train_dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(dev_rows) if dev_rows else None

        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        sft_args = SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=config.per_device_batch_size,
            per_device_eval_batch_size=max(1, config.per_device_batch_size),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=float(config.num_epochs),
            max_steps=max_steps_override if max_steps_override is not None else (config.max_steps or -1),
            max_length=config.context_length,
            bf16=use_cuda,
            gradient_checkpointing=True,
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=max(1, (max_steps_override if max_steps_override is not None else (config.max_steps or 25))),
            save_total_limit=1,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=max(1, ((max_steps_override if max_steps_override is not None else (config.max_steps or 25)) // 2)),
            report_to="none",
            dataset_text_field="text",
            do_train=True,
            do_eval=eval_dataset is not None,
            seed=42,
        )
        _emit_training_stage(
            "trainer_init_begin",
            train_example_count=len(train_rows),
            dev_example_count=len(dev_rows),
            max_steps=sft_args.max_steps,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=_build_peft_config(config.adapter_mode),
        )
        _emit_training_stage("trainer_init_done")
        _emit_training_stage("train_begin", max_steps=sft_args.max_steps)
        train_result = trainer.train()
        _emit_training_stage("train_done")
        trainer.save_model(str(out_dir / "adapter"))
        tokenizer.save_pretrained(out_dir / "tokenizer")

        if use_cuda:
            torch.cuda.synchronize()

        metrics = dict(train_result.metrics)
        payload.update(
            {
                "status": "ok",
                "finished_at": datetime.now(tz=UTC).isoformat(),
                "max_steps_effective": sft_args.max_steps,
                "train_metrics": metrics,
                "log_history": list(trainer.state.log_history),
                "peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2) if use_cuda else 0.0,
                "peak_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2) if use_cuda else 0.0,
                "output_dir_bytes": _dir_size_bytes(out_dir),
                "adapter_dir": str((out_dir / "adapter").relative_to(root)),
                "tokenizer_dir": str((out_dir / "tokenizer").relative_to(root)),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "blocked",
                "finished_at": datetime.now(tz=UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _emit_training_stage("train_blocked", error_type=type(exc).__name__, error_message=str(exc))
        try:
            import torch

            if torch.cuda.is_available():
                payload["peak_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
                payload["peak_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024**2), 2)
        except Exception:
            pass

    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def write_sft_smoke_report(root: Path, config_path: Path, out_path: Path) -> Path:
    config = load_training_config(config_path)
    payload = validate_sft_training_config(root, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
