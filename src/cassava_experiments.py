"""
cassava_experiments.py

Train and evaluate Cassava CNN experiments using the artifacts created by
cassava_eda_cleaning.py.

Experiment 1 compares three architectures:
- baseline_cnn
- residual_cnn
- residual_attention_cnn

Experiment 2 retrains the winning Experiment 1 architecture under three
training setups:
- setup_a: basic horizontal-flip augmentation
- setup_b: stronger field-robust augmentation
- setup_c: setup_b plus class-weighted cross-entropy
"""

## 0.1 Imports

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


## 0.2 Constants

SEED = 42
PROJECT_NAME = "cass_proj_csci5922"
MODEL_NAMES = ["baseline_cnn", "residual_cnn", "residual_attention_cnn"]
SETUP_NAMES = ["setup_a", "setup_b", "setup_c"]


## 0.3 Reproducibility

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## 1.1 Project paths

def get_paths(project_name: str, use_drive: bool) -> Dict[str, Path]:
    base_runtime_dir = Path("/content") / project_name
    extracted_data_dir = base_runtime_dir / "extracted_data"

    if use_drive:
        artifact_dir = Path("/content/drive/MyDrive") / project_name / "artifacts"
    else:
        artifact_dir = base_runtime_dir / "artifacts"

    model_output_dir = artifact_dir / "notebook2_outputs"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "extracted_data_dir": extracted_data_dir,
        "artifact_dir": artifact_dir,
        "model_output_dir": model_output_dir,
    }


## 1.2 Optional Google Drive mount

def maybe_mount_drive(use_drive: bool) -> None:
    if not use_drive:
        return

    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except ModuleNotFoundError:
        raise RuntimeError("Google Drive mounting requires a Colab runtime.")


## 2.1 Load saved artifacts

def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_artifacts(artifact_dir: Path, extracted_data_dir: Path) -> Dict[str, object]:
    clean_df = pd.read_csv(artifact_dir / "clean_df.csv")
    train_df = pd.read_csv(artifact_dir / "train_split_df.csv")
    val_df = pd.read_csv(artifact_dir / "val_split_df.csv")
    test_df = pd.read_csv(artifact_dir / "test_split_df.csv")

    label_mapping = load_json(artifact_dir / "label_mapping.json")
    preprocessing_config = load_json(artifact_dir / "preprocessing_config.json")
    pipeline_decisions = load_json(artifact_dir / "pipeline_decisions.json")

    label_to_name = {int(k): v for k, v in label_mapping["label_to_name"].items()}

    train_images_dir = extracted_data_dir / "train_images"
    for df in [clean_df, train_df, val_df, test_df]:
        df["image_path"] = df["image_id"].apply(lambda image_id: train_images_dir / image_id)
        df["class_name"] = df["label"].map(label_to_name)

    return {
        "clean_df": clean_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "label_to_name": label_to_name,
        "preprocessing_config": preprocessing_config,
        "pipeline_decisions": pipeline_decisions,
    }


## 3.1 Dataset wrapper

class CassavaImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")

        return self.transform(image), int(row["label"])


## 3.2 Transform definitions

def build_transforms(image_size: int, train_mean: list[float], train_std: list[float]) -> Dict[str, Callable]:
    clean_eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    setup_a_train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    setup_b_train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    return {
        "exp1_train": clean_eval_transform,
        "eval": clean_eval_transform,
        "setup_a_train": setup_a_train_transform,
        "setup_b_train": setup_b_train_transform,
        "setup_c_train": setup_b_train_transform,
    }


## 3.3 Dataloaders

def make_loader(df: pd.DataFrame, transform: Callable, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = CassavaImageDataset(df, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    transform_dict: Dict[str, Callable],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    return {
        "exp1_train": make_loader(train_df, transform_dict["exp1_train"], batch_size, True, num_workers),
        "exp1_val": make_loader(val_df, transform_dict["eval"], batch_size, False, num_workers),
        "exp1_test": make_loader(test_df, transform_dict["eval"], batch_size, False, num_workers),
        "setup_a_train": make_loader(train_df, transform_dict["setup_a_train"], batch_size, True, num_workers),
        "setup_b_train": make_loader(train_df, transform_dict["setup_b_train"], batch_size, True, num_workers),
        "setup_c_train": make_loader(train_df, transform_dict["setup_c_train"], batch_size, True, num_workers),
        "exp2_val": make_loader(val_df, transform_dict["eval"], batch_size, False, num_workers),
        "exp2_test": make_loader(test_df, transform_dict["eval"], batch_size, False, num_workers),
    }


## 4.1 Metric helpers

def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> Dict[str, object]:
    labels = list(range(num_classes))
    y_true = list(y_true)
    y_pred = list(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def per_class_recall_to_df(per_class_recall: list[float], label_to_name: Dict[int, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "label": list(range(len(per_class_recall))),
        "class_name": [label_to_name[i] for i in range(len(per_class_recall))],
        "recall": per_class_recall,
    })


def confusion_matrix_to_df(cm: np.ndarray, label_to_name: Dict[int, str]) -> pd.DataFrame:
    labels = [label_to_name[i] for i in range(cm.shape[0])]
    return pd.DataFrame(cm, index=labels, columns=labels)


## 5.1 Training loop

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_classes: int) -> Dict[str, object]:
    model.train()

    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_predictions = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        predictions = torch.argmax(logits, dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_predictions, num_classes)
    metrics["loss"] = float(total_loss / max(total_examples, 1))

    return metrics


@torch.no_grad()
def evaluate_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> Dict[str, object]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_predictions = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        predictions = torch.argmax(logits, dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_predictions, num_classes)
    metrics["loss"] = float(total_loss / max(total_examples, 1))

    return metrics


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_training_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    num_classes: int,
    num_epochs: int,
    experiment_name: str,
) -> Dict[str, object]:
    model = model.to(device)

    history = []
    best_state_dict = None
    best_epoch = None
    best_val_macro_f1 = -np.inf
    best_val_accuracy = -np.inf

    for epoch in range(num_epochs):
        start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_metrics = evaluate_one_epoch(model, val_loader, criterion, device, num_classes)

        scheduler.step(float(val_metrics["macro_f1"]))
        epoch_seconds = time.time() - start_time

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "learning_rate": current_lr(optimizer),
            "epoch_seconds": float(epoch_seconds),
        }
        history.append(row)

        is_better_macro_f1 = row["val_macro_f1"] > best_val_macro_f1
        is_tied_but_better_acc = np.isclose(row["val_macro_f1"], best_val_macro_f1) and row["val_accuracy"] > best_val_accuracy

        if is_better_macro_f1 or is_tied_but_better_acc:
            best_val_macro_f1 = row["val_macro_f1"]
            best_val_accuracy = row["val_accuracy"]
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"{experiment_name} | epoch {epoch + 1:02d}/{num_epochs} | "
            f"train_macro_f1={row['train_macro_f1']:.4f} | "
            f"val_macro_f1={row['val_macro_f1']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"lr={row['learning_rate']:.6f}"
        )

    return {
        "model": model,
        "history": history,
        "best_state_dict": best_state_dict,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_val_accuracy": float(best_val_accuracy),
    }


## 6.1 Model definitions

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, start_dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity

        return self.relu(out)


class ResidualCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResidualBlock(32, 32), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(ResidualBlock(32, 64), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(ResidualBlock(64, 128), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(ResidualBlock(128, 256), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(torch.flatten(x, start_dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mlp(self.pool(x))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.channel_attention = ChannelAttention(out_channels, reduction=reduction)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.channel_attention(self.bn2(self.conv2(out)))
        out = out + identity

        return self.relu(out)


class ResidualAttentionCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResidualAttentionBlock(32, 32), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(ResidualAttentionBlock(32, 64), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(ResidualAttentionBlock(64, 128), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(ResidualAttentionBlock(128, 256), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(torch.flatten(x, start_dim=1))


## 6.2 Model factory

def create_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if model_name == "residual_cnn":
        return ResidualCNN(num_classes)
    if model_name == "residual_attention_cnn":
        return ResidualAttentionCNN(num_classes)

    raise ValueError(f"unknown model name: {model_name}")


## 7.1 Optimizer helpers

def create_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


## 8.1 Experiment 1

def run_experiment_1(loaders: Dict[str, DataLoader], num_classes: int, args: argparse.Namespace, device: torch.device, label_to_name: Dict[int, str]) -> Dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    model_results = {}
    summary_rows = {}
    best_val_results = {}

    for model_name in MODEL_NAMES:
        model = create_model(model_name, num_classes)
        optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
        scheduler = create_scheduler(optimizer)

        result = run_training_experiment(
            model=model,
            train_loader=loaders["exp1_train"],
            val_loader=loaders["exp1_val"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_classes=num_classes,
            num_epochs=args.epochs,
            experiment_name=model_name,
        )

        result_model = result["model"]
        result_model.load_state_dict(result["best_state_dict"])
        best_validation_metrics = evaluate_one_epoch(result_model, loaders["exp1_val"], criterion, device, num_classes)

        model_results[model_name] = result
        best_val_results[model_name] = best_validation_metrics
        summary_rows[model_name] = {
            "model_name": model_name,
            "parameter_count": count_trainable_parameters(create_model(model_name, num_classes)),
            "best_epoch": result["best_epoch"],
            "best_val_accuracy": result["best_val_accuracy"],
            "best_val_macro_f1": result["best_val_macro_f1"],
        }

    results_df = pd.DataFrame(summary_rows.values())
    ranked_df = results_df.sort_values(
        by=["best_val_macro_f1", "best_val_accuracy", "parameter_count"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_model_name = str(ranked_df.loc[0, "model_name"])
    best_model = create_model(best_model_name, num_classes)
    best_model.load_state_dict(model_results[best_model_name]["best_state_dict"])
    best_model = best_model.to(device)
    test_metrics = evaluate_one_epoch(best_model, loaders["exp1_test"], criterion, device, num_classes)

    return {
        "results_df": results_df,
        "ranked_df": ranked_df,
        "model_results": model_results,
        "best_val_results": best_val_results,
        "best_model_name": best_model_name,
        "best_model_state_dict": model_results[best_model_name]["best_state_dict"],
        "best_parameter_count": int(ranked_df.loc[0, "parameter_count"]),
        "test_metrics": test_metrics,
        "criterion": criterion,
    }


## 8.2 Experiment 2

def build_class_weights(train_df: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = train_df["label"].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=0)
    count_tensor = torch.tensor(counts.values, dtype=torch.float32)

    weights = count_tensor.sum() / (num_classes * count_tensor.clamp(min=1.0))
    weights = weights / weights.mean()

    return weights.to(device)


def run_experiment_2(
    loaders: Dict[str, DataLoader],
    train_df: pd.DataFrame,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    exp1_best_model_name: str,
) -> Dict[str, object]:
    class_weights = build_class_weights(train_df, num_classes, device)

    setup_specs = {
        "setup_a": {
            "loader": loaders["setup_a_train"],
            "criterion": nn.CrossEntropyLoss(),
        },
        "setup_b": {
            "loader": loaders["setup_b_train"],
            "criterion": nn.CrossEntropyLoss(),
        },
        "setup_c": {
            "loader": loaders["setup_c_train"],
            "criterion": nn.CrossEntropyLoss(weight=class_weights),
        },
    }

    setup_results = {}
    best_val_results = {}
    rows = []

    for setup_name, spec in setup_specs.items():
        model = create_model(exp1_best_model_name, num_classes)
        optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
        scheduler = create_scheduler(optimizer)

        result = run_training_experiment(
            model=model,
            train_loader=spec["loader"],
            val_loader=loaders["exp2_val"],
            criterion=spec["criterion"],
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_classes=num_classes,
            num_epochs=args.epochs,
            experiment_name=setup_name,
        )

        result_model = result["model"]
        result_model.load_state_dict(result["best_state_dict"])
        best_validation_metrics = evaluate_one_epoch(result_model, loaders["exp2_val"], spec["criterion"], device, num_classes)

        setup_results[setup_name] = result
        best_val_results[setup_name] = best_validation_metrics
        rows.append({
            "setup_name": setup_name,
            "best_epoch": result["best_epoch"],
            "best_val_accuracy": result["best_val_accuracy"],
            "best_val_macro_f1": result["best_val_macro_f1"],
        })

    results_df = pd.DataFrame(rows)
    ranked_df = results_df.sort_values(
        by=["best_val_macro_f1", "best_val_accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_setup_name = str(ranked_df.loc[0, "setup_name"])
    best_model = create_model(exp1_best_model_name, num_classes)
    best_model.load_state_dict(setup_results[best_setup_name]["best_state_dict"])
    best_model = best_model.to(device)
    test_metrics = evaluate_one_epoch(
        model=best_model,
        dataloader=loaders["exp2_test"],
        criterion=setup_specs[best_setup_name]["criterion"],
        device=device,
        num_classes=num_classes,
    )

    return {
        "results_df": results_df,
        "ranked_df": ranked_df,
        "setup_results": setup_results,
        "best_val_results": best_val_results,
        "best_setup_name": best_setup_name,
        "best_model_state_dict": setup_results[best_setup_name]["best_state_dict"],
        "test_metrics": test_metrics,
        "class_weights": class_weights.detach().cpu().tolist(),
    }


## 9.1 Save outputs

def json_safe_history(results: Dict[str, object], keys: Iterable[str]) -> Dict[str, list[dict]]:
    return {key: results[key]["history"] for key in keys}


def save_outputs(
    output_dir: Path,
    exp1: Dict[str, object],
    exp2: Dict[str, object],
    label_to_name: Dict[int, str],
    preprocessing_config: Dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(exp1["best_model_state_dict"], output_dir / "exp1_best_model.pt")
    torch.save(exp2["best_model_state_dict"], output_dir / "exp2_best_model.pt")

    exp1["results_df"].to_csv(output_dir / "exp1_results.csv", index=False)
    exp2["results_df"].to_csv(output_dir / "exp2_results.csv", index=False)

    with open(output_dir / "exp1_history.json", "w") as f:
        json.dump(json_safe_history(exp1["model_results"], MODEL_NAMES), f, indent=2)

    with open(output_dir / "exp2_history.json", "w") as f:
        json.dump(json_safe_history(exp2["setup_results"], SETUP_NAMES), f, indent=2)

    exp1_test = exp1["test_metrics"]
    exp2_test = exp2["test_metrics"]

    final_test_metrics = {
        "experiment_1": {
            "winning_model": exp1["best_model_name"],
            "parameter_count": int(exp1["best_parameter_count"]),
            "test_accuracy": float(exp1_test["accuracy"]),
            "test_macro_f1": float(exp1_test["macro_f1"]),
            "test_loss": float(exp1_test["loss"]),
        },
        "experiment_2": {
            "winning_setup": exp2["best_setup_name"],
            "test_accuracy": float(exp2_test["accuracy"]),
            "test_macro_f1": float(exp2_test["macro_f1"]),
            "test_loss": float(exp2_test["loss"]),
        },
    }

    with open(output_dir / "final_test_metrics.json", "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    confusion_matrices = {
        "experiment_1": {
            "winning_model": exp1["best_model_name"],
            "confusion_matrix": exp1_test["confusion_matrix"].tolist(),
        },
        "experiment_2": {
            "winning_setup": exp2["best_setup_name"],
            "confusion_matrix": exp2_test["confusion_matrix"].tolist(),
        },
    }

    with open(output_dir / "confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    exp1_recall_df = per_class_recall_to_df(exp1_test["per_class_recall"], label_to_name)
    exp2_recall_df = per_class_recall_to_df(exp2_test["per_class_recall"], label_to_name)

    exp1_recall_df.to_csv(output_dir / "exp1_final_per_class_recall.csv", index=False)
    exp2_recall_df.to_csv(output_dir / "exp2_final_per_class_recall.csv", index=False)

    notebook2_config = {
        "project_name": args.project_name,
        "seed": args.seed,
        "device_at_save_time": str(device),
        "final_input_size": preprocessing_config["final_input_size"],
        "final_resize_policy": preprocessing_config["final_resize_policy"],
        "final_normalization": preprocessing_config["final_normalization"],
        "train_mean": preprocessing_config["train_mean"],
        "train_std": preprocessing_config["train_std"],
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "primary_selection_metric": "validation_macro_f1",
        "secondary_metrics": ["validation_accuracy", "parameter_count"],
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "experiment_1": {
            "winning_model": exp1["best_model_name"],
            "parameter_count": int(exp1["best_parameter_count"]),
        },
        "experiment_2": {
            "winning_setup": exp2["best_setup_name"],
            "setup_c_class_weights": [float(x) for x in exp2["class_weights"]],
        },
    }

    with open(output_dir / "notebook2_config.json", "w") as f:
        json.dump(notebook2_config, f, indent=2)


## 10.1 Main workflow

def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    maybe_mount_drive(args.use_drive)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    paths = get_paths(args.project_name, args.use_drive)

    artifacts = load_artifacts(paths["artifact_dir"], paths["extracted_data_dir"])

    preprocessing_config = artifacts["preprocessing_config"]
    label_to_name = artifacts["label_to_name"]
    num_classes = len(label_to_name)

    transforms_dict = build_transforms(
        image_size=int(preprocessing_config["final_input_size"]),
        train_mean=preprocessing_config["train_mean"],
        train_std=preprocessing_config["train_std"],
    )

    loaders = build_loaders(
        train_df=artifacts["train_df"],
        val_df=artifacts["val_df"],
        test_df=artifacts["test_df"],
        transform_dict=transforms_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    exp1 = run_experiment_1(
        loaders=loaders,
        num_classes=num_classes,
        args=args,
        device=device,
        label_to_name=label_to_name,
    )

    exp2 = run_experiment_2(
        loaders=loaders,
        train_df=artifacts["train_df"],
        num_classes=num_classes,
        args=args,
        device=device,
        exp1_best_model_name=exp1["best_model_name"],
    )

    save_outputs(
        output_dir=paths["model_output_dir"],
        exp1=exp1,
        exp2=exp2,
        label_to_name=label_to_name,
        preprocessing_config=preprocessing_config,
        args=args,
        device=device,
    )

    print(f"Experiment 1 winner: {exp1['best_model_name']}")
    print(f"Experiment 2 winner: {exp2['best_setup_name']}")
    print(f"saved model outputs to {paths['model_output_dir']}")


## 10.2 CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run cassava cnn architecture and training-strategy experiments")
    parser.add_argument("--project-name", default=PROJECT_NAME)
    parser.add_argument("--use-drive", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
