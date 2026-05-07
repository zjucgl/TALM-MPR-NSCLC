import argparse
import concurrent.futures
import copy
import glob
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "train" / "config.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.network import TemporalAlignedNet


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def load_config(config_path):
    config_path = resolve_path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(config_path)
    return cfg


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, targets, weights=None):
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if weights is not None:
            loss = loss * weights.view(-1, 1)
        return loss.mean()


def aligned_collate(batch):
    def pad_and_mask(data_key):
        sequences = [item[data_key] for item in batch]
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True)
        batch_size, max_steps = len(sequences), padded.size(1)
        mask = torch.arange(max_steps).expand(batch_size, max_steps) < lengths.unsqueeze(1)
        return padded, mask

    def pad_bool(data_key):
        return pad_sequence([item[data_key] for item in batch], batch_first=True, padding_value=False)

    def pad_float(data_key):
        return pad_sequence([item[data_key] for item in batch], batch_first=True, padding_value=0.0)

    m1_roi, global_mask = pad_and_mask("m1_roi")
    m1_ctx, _ = pad_and_mask("m1_ctx")
    m2, _ = pad_and_mask("m2")
    m3, _ = pad_and_mask("m3")

    return {
        "pids": [item["pid"] for item in batch],
        "m1_roi": m1_roi,
        "m1_ctx": m1_ctx,
        "m2": m2,
        "m3": m3,
        "global_mask": global_mask,
        "m1_avail": pad_bool("m1_avail"),
        "m2_avail": pad_bool("m2_avail"),
        "m3_avail": pad_bool("m3_avail"),
        "time_deltas": pad_float("time_deltas"),
        "is_temporal": torch.tensor([item["num_windows"] > 1 for item in batch], dtype=torch.bool),
        "labels": {
            "mpr": torch.stack([item["label_mpr"] for item in batch]),
        },
    }


class TimeWindowLongitudinalDataset(Dataset):
    def __init__(self, cfg, split="all"):
        self.cfg = cfg
        self.split = split

        data_cfg = cfg["data"]
        path_cfg = cfg["paths"]
        dims_cfg = cfg["model_dims"]

        self.window_days = data_cfg["window_days"]
        self.time_scale_days = data_cfg["time_scale_days"]
        self.max_time_delta = data_cfg["max_time_delta"]
        self.input_root = path_cfg["input_root"].replace("\\", "/").rstrip("/")
        self.tensor_root = resolve_path(path_cfg["tensor_root"])
        self.tensor_suffixes = data_cfg["tensor_suffixes"]
        self.roi_shape = tuple(data_cfg["missing_tensor_shapes"]["roi"])
        self.context_shape = tuple(data_cfg["missing_tensor_shapes"]["context"])
        self.m2_input_dim = dims_cfg["m2_input_dim"]
        self.m3_input_dim = dims_cfg["m3_input_dim"]

        norm_path = self.tensor_root / path_cfg["m2_norm_params_file"]
        if norm_path.exists():
            params = torch.load(norm_path, weights_only=True)
            self.m2_mean, self.m2_std = params["mean"], params["std"]
        elif data_cfg.get("allow_missing_m2_norm", False):
            self.m2_mean = torch.zeros(self.m2_input_dim)
            self.m2_std = torch.ones(self.m2_input_dim)
        else:
            raise FileNotFoundError(
                f"Missing radiomics normalization file: {norm_path}. "
                "Create it during preprocessing or set data.allow_missing_m2_norm=true for schema-only tests."
            )

        self.labels_path = resolve_path(path_cfg["labels_json"])
        self.m1_m2_path = resolve_path(path_cfg["m1_m2_json"])
        self.m3_path = resolve_path(path_cfg["m3_json"])

        self.patients = self._build_registry()
        self.pids = list(self.patients.keys())

    def _tensor_path(self, raw_path, suffix):
        raw = raw_path.replace("\\", "/")
        if raw.startswith(self.input_root + "/"):
            rel = raw[len(self.input_root) + 1 :]
        else:
            rel = raw.lstrip("/")
        return self.tensor_root / Path(PurePosixPath(rel + suffix))

    @staticmethod
    def _parse_date(date_str):
        return datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")

    @staticmethod
    def _load_json(path):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_registry(self):
        labels = self._load_json(self.labels_path)
        m1_m2_lookup = {item["reg_no"]: item["time_steps"] for item in self._load_json(self.m1_m2_path)}
        m3_lookup = {item["reg_no"]: item["time_steps"] for item in self._load_json(self.m3_path)}

        registry = {}
        for label_item in labels:
            pid = label_item["pat_id"]
            raw_records = []

            for reg_no in label_item["reg_no"]:
                for step in m1_m2_lookup.get(reg_no, []):
                    record = copy.deepcopy(step)
                    record["source"] = "m1m2"
                    record["dt"] = self._parse_date(record["date"])
                    raw_records.append(record)

                for step in m3_lookup.get(reg_no, []):
                    record = copy.deepcopy(step)
                    record["source"] = "m3"
                    record["dt"] = self._parse_date(record["date"])
                    raw_records.append(record)

            if not raw_records:
                continue

            raw_records.sort(key=lambda record: record["dt"])
            windows = self._group_records_into_windows(raw_records)
            valid_windows = [window for window in windows if window["m1m2"] is not None or window["m3"] is not None]

            if valid_windows:
                registry[pid] = {
                    "windows": valid_windows,
                    "label": float(label_item["MPR"]),
                }

        return registry

    def _group_records_into_windows(self, records):
        windows = []
        window_start = None
        current_window = {"m1m2": None, "m3": None}

        for record in records:
            if window_start is None or (record["dt"] - window_start).days > self.window_days:
                if window_start is not None:
                    current_window["window_start_dt"] = window_start
                    windows.append(current_window)
                window_start = record["dt"]
                current_window = {"m1m2": None, "m3": None}

            if current_window[record["source"]] is not None:
                continue

            if record["source"] == "m1m2":
                roi_path = self._tensor_path(record["file_path"], self.tensor_suffixes["roi"])
                ctx_path = self._tensor_path(record["file_path"], self.tensor_suffixes["context"])
                rad_path = self._tensor_path(record["file_path"], self.tensor_suffixes["radiomics"])
                record["has_m1"] = roi_path.exists() and ctx_path.exists()
                record["has_m2"] = rad_path.exists()
                if record["has_m1"] or record["has_m2"] or self.cfg["data"].get("keep_missing_imaging_records", False):
                    current_window["m1m2"] = record
            else:
                current_window["m3"] = record

        if window_start is not None:
            current_window["window_start_dt"] = window_start
            windows.append(current_window)

        return windows

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        patient = self.patients[pid]

        m1_roi_seq, m1_ctx_seq, m2_seq, m3_seq = [], [], [], []
        m1_avail, m2_avail, m3_avail = [], [], []
        time_deltas = []

        baseline_dt = patient["windows"][0]["window_start_dt"]
        for window in patient["windows"]:
            delta_days = (window["window_start_dt"] - baseline_dt).days
            time_deltas.append(min(delta_days / self.time_scale_days, self.max_time_delta))

            self._append_m1_m2(window, m1_roi_seq, m1_ctx_seq, m2_seq, m1_avail, m2_avail)
            self._append_m3(window, m3_seq, m3_avail)

        return {
            "pid": pid,
            "num_windows": len(patient["windows"]),
            "m1_roi": torch.stack(m1_roi_seq),
            "m1_ctx": torch.stack(m1_ctx_seq),
            "m2": torch.stack(m2_seq),
            "m3": torch.stack(m3_seq),
            "m1_avail": torch.tensor(m1_avail, dtype=torch.bool),
            "m2_avail": torch.tensor(m2_avail, dtype=torch.bool),
            "m3_avail": torch.tensor(m3_avail, dtype=torch.bool),
            "time_deltas": torch.tensor(time_deltas, dtype=torch.float32),
            "label_mpr": torch.tensor(patient["label"], dtype=torch.float32),
        }

    def _append_m1_m2(self, window, m1_roi_seq, m1_ctx_seq, m2_seq, m1_avail, m2_avail):
        if window["m1m2"] is None:
            m1_roi_seq.append(torch.zeros(self.roi_shape))
            m1_ctx_seq.append(torch.zeros(self.context_shape))
            m2_seq.append(torch.zeros(self.m2_input_dim))
            m1_avail.append(False)
            m2_avail.append(False)
            return

        step = window["m1m2"]
        if step.get("has_m1", False):
            roi_path = self._tensor_path(step["file_path"], self.tensor_suffixes["roi"])
            ctx_path = self._tensor_path(step["file_path"], self.tensor_suffixes["context"])
            m1_roi_seq.append(torch.load(roi_path, weights_only=True))
            m1_ctx_seq.append(torch.load(ctx_path, weights_only=True))
            m1_avail.append(True)
        else:
            m1_roi_seq.append(torch.zeros(self.roi_shape))
            m1_ctx_seq.append(torch.zeros(self.context_shape))
            m1_avail.append(False)

        if step.get("has_m2", False):
            m2_path = self._tensor_path(step["file_path"], self.tensor_suffixes["radiomics"])
            m2_raw = torch.load(m2_path, weights_only=True)
            m2_seq.append((m2_raw - self.m2_mean) / (self.m2_std + 1e-8))
            m2_avail.append(True)
        else:
            m2_seq.append(torch.zeros(self.m2_input_dim))
            m2_avail.append(False)

    def _append_m3(self, window, m3_seq, m3_avail):
        if window["m3"] is not None:
            m3_seq.append(torch.tensor(window["m3"]["lab_features"], dtype=torch.float32))
            m3_avail.append(True)
        else:
            m3_seq.append(torch.zeros(self.m3_input_dim))
            m3_avail.append(False)


def make_autocast(device, enabled):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def move_batch_to_device(batch, device):
    return {
        "m1_roi": batch["m1_roi"].to(device),
        "m1_ctx": batch["m1_ctx"].to(device),
        "m2": batch["m2"].to(device),
        "m3": batch["m3"].to(device),
        "global_mask": batch["global_mask"].to(device),
        "m1_avail": batch["m1_avail"].to(device),
        "m2_avail": batch["m2_avail"].to(device),
        "m3_avail": batch["m3_avail"].to(device),
        "time_deltas": batch["time_deltas"].to(device),
        "targets": batch["labels"]["mpr"].to(device).view(-1, 1),
        "is_temporal": batch["is_temporal"].to(device),
    }


def apply_augmentation(m1_roi, m1_ctx, cfg):
    aug_cfg = cfg["augmentation"]
    if not aug_cfg.get("enabled", True):
        return m1_roi, m1_ctx

    flip_probability = aug_cfg["flip_probability"]
    for dim in aug_cfg["flip_dims"]:
        if torch.rand(1, device=m1_roi.device).item() < flip_probability:
            m1_roi = torch.flip(m1_roi, dims=[dim])
            m1_ctx = torch.flip(m1_ctx, dims=[dim])

    noise_std = aug_cfg.get("roi_noise_std", 0.0)
    if noise_std > 0:
        m1_roi = m1_roi + torch.randn_like(m1_roi) * noise_std

    return m1_roi, m1_ctx


def model_forward(model, batch_tensors):
    return model(
        batch_tensors["m1_roi"],
        batch_tensors["m1_ctx"],
        batch_tensors["m2"],
        batch_tensors["m3"],
        batch_tensors["global_mask"],
        batch_tensors["m1_avail"],
        batch_tensors["m2_avail"],
        batch_tensors["m3_avail"],
        batch_tensors["time_deltas"],
    )


def build_loaders(full_dataset, train_idx, val_idx, cfg):
    training_cfg = cfg["training"]
    seed = cfg["runtime"]["seed"]

    train_dataset = copy.copy(full_dataset)
    train_dataset.pids = [full_dataset.pids[i] for i in train_idx]
    train_dataset.split = "train"

    val_dataset = copy.copy(full_dataset)
    val_dataset.pids = [full_dataset.pids[i] for i in val_idx]
    val_dataset.split = "val"

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "batch_size": training_cfg["batch_size"],
        "collate_fn": aligned_collate,
        "num_workers": training_cfg["num_workers"],
        "pin_memory": training_cfg["pin_memory"],
    }

    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_optimizer_and_scheduler(model, cfg):
    training_cfg = cfg["training"]
    cnn_params = list(model.m1_roi_enc.parameters()) + list(model.m1_ctx_enc.parameters())
    cnn_param_ids = {id(param) for param in cnn_params}
    other_params = [param for param in model.parameters() if id(param) not in cnn_param_ids]

    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": training_cfg["learning_rate"]},
            {"params": cnn_params, "lr": training_cfg["cnn_learning_rate"]},
        ],
        weight_decay=training_cfg["weight_decay"],
    )

    scheduler_cfg = cfg["scheduler"]
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_cfg["t_0"],
        T_mult=scheduler_cfg["t_mult"],
        eta_min=scheduler_cfg["eta_min"],
    )
    return optimizer, scheduler


def set_cnn_trainable(model, trainable):
    for param in model.m1_roi_enc.parameters():
        param.requires_grad = trainable
    for param in model.m1_ctx_enc.parameters():
        param.requires_grad = trainable


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, cfg):
    model.train()
    total_loss = 0.0
    use_amp = cfg["runtime"]["amp"] and device.type == "cuda"
    temporal_weight = cfg["loss"]["temporal_sample_weight"]

    for batch in loader:
        batch_tensors = move_batch_to_device(batch, device)
        batch_tensors["m1_roi"], batch_tensors["m1_ctx"] = apply_augmentation(
            batch_tensors["m1_roi"],
            batch_tensors["m1_ctx"],
            cfg,
        )

        weights = torch.ones_like(batch_tensors["is_temporal"], dtype=torch.float32).view(-1, 1)
        weights = weights.to(device)
        weights[batch_tensors["is_temporal"].view(-1, 1)] = temporal_weight

        optimizer.zero_grad(set_to_none=True)
        with make_autocast(device, use_amp):
            logits, _ = model_forward(model, batch_tensors)
            loss = criterion(logits, batch_tensors["targets"], weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    rows = []
    targets_all = []
    probs_all = []
    use_amp = cfg["runtime"]["amp"] and device.type == "cuda"
    threshold = cfg["evaluation"]["prediction_threshold"]

    with torch.no_grad():
        for batch in loader:
            batch_tensors = move_batch_to_device(batch, device)
            weights = torch.ones_like(batch_tensors["is_temporal"], dtype=torch.float32).view(-1, 1).to(device)

            with make_autocast(device, use_amp):
                logits, _ = model_forward(model, batch_tensors)
                loss = criterion(logits, batch_tensors["targets"], weights)

            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            targets_np = batch_tensors["targets"].view(-1).cpu().numpy()
            preds_np = preds.view(-1).cpu().numpy()
            probs_np = probs.view(-1).cpu().numpy()

            targets_all.extend(targets_np)
            probs_all.extend(probs_np)

            for idx, pid in enumerate(batch["pids"]):
                rows.append(
                    {
                        "pid": pid,
                        "true": int(targets_np[idx]),
                        "pred": int(preds_np[idx]),
                        "pred_prob": float(probs_np[idx]),
                    }
                )

    try:
        auc = roc_auc_score(targets_all, probs_all)
    except ValueError:
        auc = 0.5

    preds_all = (np.array(probs_all) > threshold).astype(int)
    f1 = f1_score(targets_all, preds_all, zero_division=0)
    return {
        "loss": total_loss / max(len(loader), 1),
        "auc": auc,
        "f1": f1,
        "rows": rows,
    }


def collect_oof_predictions(model, loader, device, cfg, fold):
    model.eval()
    predictions = []
    use_amp = cfg["runtime"]["amp"] and device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            batch_tensors = move_batch_to_device(batch, device)
            with make_autocast(device, use_amp):
                logits, _ = model_forward(model, batch_tensors)

            probs = torch.sigmoid(logits).view(-1).cpu()
            targets = batch["labels"]["mpr"].view(-1)
            for idx, pid in enumerate(batch["pids"]):
                predictions.append(
                    {
                        "pid": pid,
                        "fold": fold,
                        "true_label": int(targets[idx].item()),
                        "pred_prob": float(probs[idx].item()),
                    }
                )

    return predictions


def build_stratification_labels(dataset):
    labels = []
    for pid in dataset.pids:
        label = int(dataset.patients[pid]["label"])
        has_longitudinal = len(dataset.patients[pid]["windows"]) > 1
        labels.append(f"{label}_{has_longitudinal}")
    return labels


def validate_cv_feasibility(labels, n_splits):
    counts = Counter(labels)
    smallest_group = min(counts.values()) if counts else 0
    if smallest_group < n_splits:
        raise ValueError(
            f"cross_validation.n_splits={n_splits} exceeds the smallest stratum size "
            f"({smallest_group}). Reduce n_splits or provide more training samples."
        )


def train_pooled_cv(cfg, target_fold=None):
    set_seed(cfg["runtime"]["seed"])
    output_dir = resolve_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(cfg["runtime"]["device"])

    dataset = TimeWindowLongitudinalDataset(cfg, split="all")
    all_indices = np.arange(len(dataset))
    strat_labels = build_stratification_labels(dataset)
    n_splits = cfg["cross_validation"]["n_splits"]
    validate_cv_feasibility(strat_labels, n_splits)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=cfg["cross_validation"]["shuffle"],
        random_state=cfg["runtime"]["seed"],
    )

    criterion = WeightedBCELoss(cfg["loss"]["pos_weight"]).to(device)
    fold_predictions = []

    print(f"Starting pooled CV with {len(dataset)} patients on {device}.")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(all_indices, strat_labels), 1):
        if target_fold is not None and fold != target_fold:
            continue

        print(f"\nFold {fold}/{n_splits}")
        train_loader, val_loader = build_loaders(dataset, train_idx, val_idx, cfg)

        model = TemporalAlignedNet(cfg).to(device)
        freeze_epochs = cfg["training"]["freeze_cnn_epochs"]
        set_cnn_trainable(model, trainable=(freeze_epochs <= 0))

        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
        use_amp = cfg["runtime"]["amp"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_auc = 0.0
        best_loss = float("inf")
        max_epochs = cfg["training"]["epochs"]
        epoch = 0
        best_weight_path = output_dir / cfg["outputs"]["checkpoint_pattern"].format(fold=fold)

        while epoch < max_epochs:
            if epoch == freeze_epochs:
                set_cnn_trainable(model, trainable=True)

            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, cfg)
            metrics = validate(model, val_loader, criterion, device, cfg)
            scheduler.step()

            improved = metrics["auc"] > best_auc or (metrics["auc"] == best_auc and metrics["loss"] < best_loss)
            marker = ""
            if improved:
                marker = " [best]"
                if metrics["auc"] > best_auc and max_epochs - epoch <= cfg["training"]["extension_patience"]:
                    max_epochs += cfg["training"]["extension_epochs"]
                best_auc = metrics["auc"]
                best_loss = metrics["loss"]
                torch.save(model.state_dict(), best_weight_path)

            print(
                f"Epoch {epoch + 1:03d}/{max_epochs:03d} | "
                f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
                f"val_auc={metrics['auc']:.4f} val_f1={metrics['f1']:.4f}{marker}"
            )
            epoch += 1

        model.load_state_dict(torch.load(best_weight_path, weights_only=True, map_location=device))
        fold_predictions.extend(collect_oof_predictions(model, val_loader, device, cfg, fold))

    output_name = (
        cfg["outputs"]["fold_oof_pattern"].format(fold=target_fold)
        if target_fold is not None
        else cfg["outputs"]["oof_filename"]
    )
    output_path = output_dir / output_name
    pd.DataFrame(fold_predictions).to_csv(output_path, index=False)
    print(f"Saved OOF predictions to {output_path}")


def run_parallel_folds(config_path, cfg):
    parallel_cfg = cfg["parallel"]
    output_dir = resolve_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_to_device = {int(k): str(v) for k, v in parallel_cfg["fold_to_device"].items()}
    n_splits = cfg["cross_validation"]["n_splits"]
    max_workers = parallel_cfg["max_workers"]

    def run_fold(fold_idx):
        env = os.environ.copy()
        if fold_idx in fold_to_device:
            env["CUDA_VISIBLE_DEVICES"] = fold_to_device[fold_idx]
        log_path = output_dir / f"fold_{fold_idx}_train.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--fold", str(fold_idx)]
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        return fold_idx, log_path

    print(f"Launching {n_splits} folds with up to {max_workers} workers.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_fold, fold): fold for fold in range(1, n_splits + 1)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_splits):
            fold_idx, log_path = future.result()
            print(f"Fold {fold_idx} finished. Log: {log_path}")

    fold_csvs = sorted(glob.glob(str(output_dir / cfg["outputs"]["fold_oof_pattern"].format(fold="*"))))
    if len(fold_csvs) == n_splits:
        combined = pd.concat([pd.read_csv(path) for path in fold_csvs], ignore_index=True)
        combined.sort_values(by=["fold", "pid"], inplace=True)
        combined.to_csv(output_dir / cfg["outputs"]["oof_filename"], index=False)
        print(f"Saved combined OOF predictions to {output_dir / cfg['outputs']['oof_filename']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TemporalAlignedNet for MPR prediction.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the JSON config file.")
    parser.add_argument("--fold", type=int, default=None, help="Train one fold only.")
    parser.add_argument("--parallel", action="store_true", help="Launch one subprocess per CV fold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    if args.parallel:
        run_parallel_folds(config_path, config)
    else:
        train_pooled_cv(config, target_fold=args.fold)
