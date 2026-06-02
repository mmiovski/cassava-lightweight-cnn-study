"""
cassava_reporting_plots.py

Generate final report figures from saved artifacts. This script combines the
plotting logic from cassava_code_04 and cassava_code_05 into one reusable file
with a single artifact-loading section.
"""

## 0.1 Imports

import argparse
import glob
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torchvision import transforms


## 0.2 Constants

SEED = 5922
PROJECT_NAME = "cass_proj_csci5922"

CLASS_ORDER = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

CLASS_COLORS = {
    "CBB": "#8B6F47",
    "CBSD": "#7A8450",
    "CGM": "#8FA98A",
    "CMD": "#4F6F52",
    "Healthy": "#C8A97E",
}

ARCH_COLORS = {
    "Baseline CNN": "#8B6F47",
    "Residual CNN": "#8FA98A",
    "Residual Attention CNN": "#4F6F52",
}

SETUP_COLORS = {
    "Setup A": "#4F6F52",
    "Setup B": "#7A8450",
    "Setup C": "#8B6F47",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "earthy_heatmap",
    ["#FFFFFF", "#E8DCC8", "#C8A97E", "#7A8450", "#4F6F52"],
)


## 0.3 Reproducibility

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


## 0.4 Plot style

def set_report_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#555555",
        "axes.labelcolor": "#2B2B2B",
        "xtick.color": "#2B2B2B",
        "ytick.color": "#2B2B2B",
        "text.color": "#2B2B2B",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


## 1.1 Project paths

def get_paths(project_name: str, use_drive: bool) -> Dict[str, Path]:
    base_runtime_dir = Path("/content") / project_name
    extracted_data_dir = base_runtime_dir / "extracted_data"

    if use_drive:
        artifact_dir = Path("/content/drive/MyDrive") / project_name / "artifacts"
    else:
        artifact_dir = base_runtime_dir / "artifacts"

    model_output_dir = artifact_dir / "notebook2_outputs"
    report_output_dir = artifact_dir / "notebook3_reporting_outputs"
    figure_dir = report_output_dir / "final_report_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    return {
        "artifact_dir": artifact_dir,
        "model_output_dir": model_output_dir,
        "report_output_dir": report_output_dir,
        "figure_dir": figure_dir,
        "extracted_data_dir": extracted_data_dir,
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


## 2.1 Shared helpers

def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def normalize_key(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def pretty_class_name(x: object) -> str:
    mapping = {
        "0": "CBB",
        "0_0": "CBB",
        "1": "CBSD",
        "1_0": "CBSD",
        "2": "CGM",
        "2_0": "CGM",
        "3": "CMD",
        "3_0": "CMD",
        "4": "Healthy",
        "4_0": "Healthy",
        "cbb": "CBB",
        "cassava_bacterial_blight_cbb": "CBB",
        "cassava_bacterial_blight": "CBB",
        "cbsd": "CBSD",
        "cassava_brown_streak_disease_cbsd": "CBSD",
        "cassava_brown_streak_disease": "CBSD",
        "cgm": "CGM",
        "cassava_green_mottle_cgm": "CGM",
        "cassava_green_mottle": "CGM",
        "cmd": "CMD",
        "cassava_mosaic_disease_cmd": "CMD",
        "cassava_mosaic_disease": "CMD",
        "healthy": "Healthy",
    }
    return mapping.get(normalize_key(x), str(x))


def pretty_model_name(x: object) -> str:
    mapping = {
        "baseline_cnn": "Baseline CNN",
        "baseline": "Baseline CNN",
        "residual_cnn": "Residual CNN",
        "residual": "Residual CNN",
        "residual_attention_cnn": "Residual Attention CNN",
        "residual_attn_cnn": "Residual Attention CNN",
        "residual_attention": "Residual Attention CNN",
    }
    return mapping.get(normalize_key(x), str(x))


def pretty_setup_name(x: object) -> str:
    mapping = {
        "setup_a": "Setup A",
        "a": "Setup A",
        "setup_b": "Setup B",
        "b": "Setup B",
        "setup_c": "Setup C",
        "c": "Setup C",
    }
    return mapping.get(normalize_key(x), str(x))


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized_columns = {normalize_key(col): col for col in df.columns}

    for candidate in candidates:
        candidate_key = normalize_key(candidate)
        if candidate_key in normalized_columns:
            return normalized_columns[candidate_key]

    return None


def infer_count_column(df: pd.DataFrame) -> str:
    preferred = ["count", "counts", "n", "total", "num_images"]
    col = find_column(df, preferred)
    if col is not None:
        return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("no numeric count column found")

    return numeric_cols[0]


def infer_recall_column(df: pd.DataFrame) -> str:
    preferred = ["recall", "per_class_recall", "value", "score"]
    col = find_column(df, preferred)
    if col is not None:
        return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("no numeric recall column found")

    return numeric_cols[0]


def save_figure(fig: plt.Figure, figure_dir: Path, filename: str) -> None:
    png_path = figure_dir / filename
    pdf_path = png_path.with_suffix(".pdf")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


## 2.2 Path resolution

def resolve_path(search_dirs: Iterable[Path], patterns: Iterable[str]) -> Path:
    for folder in search_dirs:
        for pattern in patterns:
            matches = sorted(glob.glob(str(folder / pattern)))
            if matches:
                return Path(matches[0])

    raise FileNotFoundError(f"could not resolve any pattern: {list(patterns)}")


def resolve_artifact_paths(paths: Dict[str, Path]) -> Dict[str, Path]:
    search_dirs = [
        paths["report_output_dir"],
        paths["model_output_dir"],
        paths["artifact_dir"],
    ]

    return {
        "dataset_class_distribution": resolve_path(search_dirs, ["dataset_class_distribution.csv", "class_count_summary.csv"]),
        "exp1_history": resolve_path(search_dirs, ["exp1_history.json"]),
        "exp2_history": resolve_path(search_dirs, ["exp2_history.json"]),
        "confusion_matrices": resolve_path(search_dirs, ["confusion_matrices.json"]),
        "exp2_final_per_class_recall": resolve_path(search_dirs, ["exp2_final_per_class_recall.csv", "exp2_final_per_class_recall(1).csv"]),
        "train_split": resolve_path(search_dirs, ["train_split_df.csv"]),
        "val_split": resolve_path(search_dirs, ["val_split_df.csv"]),
        "test_split": resolve_path(search_dirs, ["test_split_df.csv"]),
        "preprocessing_config": resolve_path(search_dirs, ["preprocessing_config.json"]),
        "label_mapping": resolve_path(search_dirs, ["label_mapping.json"]),
    }


## 3.1 Load plotting artifacts once

def load_plotting_artifacts(paths: Dict[str, Path]) -> Dict[str, object]:
    artifact_paths = resolve_artifact_paths(paths)

    label_mapping = load_json(artifact_paths["label_mapping"])
    preprocessing_config = load_json(artifact_paths["preprocessing_config"])

    label_to_name = {int(k): v for k, v in label_mapping["label_to_name"].items()}

    return {
        "paths": artifact_paths,
        "label_to_name": label_to_name,
        "preprocessing_config": preprocessing_config,
        "class_distribution": pd.read_csv(artifact_paths["dataset_class_distribution"]),
        "exp1_history": load_json(artifact_paths["exp1_history"]),
        "exp2_history": load_json(artifact_paths["exp2_history"]),
        "confusion_matrices": load_json(artifact_paths["confusion_matrices"]),
        "exp2_final_per_class_recall": pd.read_csv(artifact_paths["exp2_final_per_class_recall"]),
        "train_split": pd.read_csv(artifact_paths["train_split"]),
        "val_split": pd.read_csv(artifact_paths["val_split"]),
        "test_split": pd.read_csv(artifact_paths["test_split"]),
    }


## 4.1 Tidy class distribution

def tidy_class_distribution_df(df: pd.DataFrame) -> pd.DataFrame:
    class_col = find_column(df, ["class_abbreviation", "class", "class_name", "label", "disease", "category"])
    count_col = infer_count_column(df)

    if class_col is None:
        raise ValueError("could not find class column in class distribution file")

    out = df[[class_col, count_col]].copy()
    out.columns = ["class", "count"]
    out["class"] = out["class"].map(pretty_class_name)
    out = out.groupby("class", as_index=False)["count"].sum()
    out["class"] = pd.Categorical(out["class"], categories=CLASS_ORDER, ordered=True)

    return out.sort_values("class").reset_index(drop=True)


## 4.2 Tidy recall

def tidy_recall_df(df: pd.DataFrame) -> pd.DataFrame:
    class_col = find_column(df, ["class", "class_name", "label", "disease", "category"])
    recall_col = infer_recall_column(df)

    if class_col is None:
        raise ValueError("could not find class column in recall file")

    out = df[[class_col, recall_col]].copy()
    out.columns = ["class", "recall"]
    out["class"] = out["class"].map(pretty_class_name)
    out["recall"] = pd.to_numeric(out["recall"], errors="coerce")

    if out["recall"].max() > 1.05:
        out["recall"] = out["recall"] / 100.0

    out = out.dropna(subset=["recall"])
    out = out[out["class"].isin(CLASS_ORDER)].copy()
    out = out.groupby("class", as_index=False)["recall"].mean()
    out["class"] = pd.Categorical(out["class"], categories=CLASS_ORDER, ordered=True)

    return out.sort_values("class").reset_index(drop=True)


## 4.3 Training-history helpers

def extract_train_macro_f1_series(history_json: Dict, key_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    series = {}

    for raw_key, display_name in key_map.items():
        if raw_key not in history_json:
            raise KeyError(f"{raw_key} not found in history json")

        rows = history_json[raw_key]
        series[display_name] = np.array([row["train_macro_f1"] for row in rows], dtype=float)

    return series


## 5.1 Figure 1 class distribution

def plot_class_distribution(artifacts: Dict[str, object], figure_dir: Path) -> None:
    df_class = tidy_class_distribution_df(artifacts["class_distribution"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_colors = [CLASS_COLORS[c] for c in df_class["class"]]
    bars = ax.bar(df_class["class"].astype(str), df_class["count"], color=bar_colors, edgecolor="black", linewidth=0.6)

    ax.set_title("Class Distribution in the Dataset")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_ylim(0, df_class["count"].max() * 1.12)
    ax.grid(axis="y", alpha=0.2)

    for bar, count in zip(bars, df_class["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(df_class["count"]) * 0.01,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    save_figure(fig, figure_dir, "figure_1_class_distribution.png")


## 5.2 Figure 2 preprocessing and augmentation examples

def add_runtime_image_paths(df: pd.DataFrame, train_images_dir: Path) -> pd.DataFrame:
    out = df.copy()
    out["image_path"] = out["image_id"].apply(lambda image_id: train_images_dir / image_id)
    return out


def pil_loader(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def inverse_normalize(img_tensor: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    img = img_tensor.clone().cpu()
    img = img * std_tensor + mean_tensor
    img = torch.clamp(img, 0.0, 1.0)

    return img.permute(1, 2, 0).numpy()


def make_reporting_transforms(image_size: int, train_mean: list[float], train_std: list[float]) -> Dict[str, transforms.Compose]:
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    setup_a_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    setup_b_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    return {
        "base": base_transform,
        "setup_a": setup_a_transform,
        "setup_b": setup_b_transform,
    }


def apply_and_plot_ready(pil_img: Image.Image, transform: transforms.Compose, mean: list[float], std: list[float]) -> np.ndarray:
    return inverse_normalize(transform(pil_img), mean, std)


def resized_np_from_pil(pil_img: Image.Image, image_size: int) -> np.ndarray:
    img = pil_img.resize((image_size, image_size))
    return np.asarray(img).astype(np.float32) / 255.0


def flipped_np_from_pil(pil_img: Image.Image, image_size: int) -> np.ndarray:
    img = pil_img.resize((image_size, image_size)).transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(img).astype(np.float32) / 255.0


def find_setup_a_flip_seed(
    pil_img: Image.Image,
    setup_a_transform: transforms.Compose,
    image_size: int,
    train_mean: list[float],
    train_std: list[float],
    max_tries: int = 300,
) -> int | None:
    base_np = resized_np_from_pil(pil_img, image_size)
    flip_np = flipped_np_from_pil(pil_img, image_size)

    for seed in range(max_tries):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        aug_np = apply_and_plot_ready(pil_img, setup_a_transform, train_mean, train_std)
        base_err = np.mean((aug_np - base_np) ** 2)
        flip_err = np.mean((aug_np - flip_np) ** 2)

        if flip_err < base_err:
            return seed

    return None


def build_candidate_pool(df: pd.DataFrame, seed: int) -> list[dict]:
    candidate_rows = []

    if "label" in df.columns:
        for label in sorted(df["label"].dropna().unique()):
            subset = df[df["label"] == label].sample(frac=1, random_state=seed)
            candidate_rows.extend(row.to_dict() for _, row in subset.iterrows())
    else:
        candidate_rows.extend(row.to_dict() for _, row in df.sample(frac=1, random_state=seed).iterrows())

    return candidate_rows


def select_augmentation_examples(
    val_df: pd.DataFrame,
    setup_a_transform: transforms.Compose,
    image_size: int,
    train_mean: list[float],
    train_std: list[float],
    seed: int,
) -> list[dict]:
    candidate_pool = build_candidate_pool(val_df, seed=seed)
    selected_rows = []
    selected_ids = set()
    flipped_count = 0

    for row in candidate_pool:
        if row["image_id"] in selected_ids:
            continue

        pil_img = pil_loader(row["image_path"])
        flip_seed = find_setup_a_flip_seed(pil_img, setup_a_transform, image_size, train_mean, train_std)

        if flip_seed is not None:
            row["setup_a_seed"] = int(flip_seed)
            selected_rows.append(row)
            selected_ids.add(row["image_id"])
            flipped_count += 1

        if flipped_count >= 2:
            break

    if flipped_count < 2:
        raise ValueError("could not find at least two samples with visible setup A flips")

    for row in candidate_pool:
        if len(selected_rows) >= 3:
            break
        if row["image_id"] in selected_ids:
            continue

        pil_img = pil_loader(row["image_path"])
        flip_seed = find_setup_a_flip_seed(pil_img, setup_a_transform, image_size, train_mean, train_std)
        row["setup_a_seed"] = int(flip_seed) if flip_seed is not None else int(seed + 1000 + len(selected_rows))

        selected_rows.append(row)
        selected_ids.add(row["image_id"])

    if len(selected_rows) < 3:
        raise ValueError("could not collect three augmentation examples")

    return selected_rows


def plot_preprocessing_augmentation_examples(artifacts: Dict[str, object], paths: Dict[str, Path], figure_dir: Path, seed: int) -> None:
    train_images_dir = paths["extracted_data_dir"] / "train_images"
    if not train_images_dir.exists():
        raise FileNotFoundError(f"train image directory not found: {train_images_dir}")

    preprocessing_config = artifacts["preprocessing_config"]
    image_size = int(preprocessing_config["final_input_size"])
    train_mean = preprocessing_config["train_mean"]
    train_std = preprocessing_config["train_std"]

    val_df = add_runtime_image_paths(artifacts["val_split"], train_images_dir)
    missing_images = [path for path in val_df["image_path"].head(10) if not Path(path).exists()]
    if missing_images:
        raise FileNotFoundError(f"runtime images are missing, first missing path: {missing_images[0]}")

    transform_dict = make_reporting_transforms(image_size, train_mean, train_std)
    selected_rows = select_augmentation_examples(
        val_df=val_df,
        setup_a_transform=transform_dict["setup_a"],
        image_size=image_size,
        train_mean=train_mean,
        train_std=train_std,
        seed=seed,
    )

    column_titles = ["Original Image", "Base Preprocessing", "Setup A", "Setup B"]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13.5, 10.0))

    for row_idx, row in enumerate(selected_rows):
        pil_img = pil_loader(row["image_path"])

        original_img = np.array(pil_img.resize((image_size, image_size)))
        base_img = apply_and_plot_ready(pil_img, transform_dict["base"], train_mean, train_std)

        random.seed(row["setup_a_seed"])
        np.random.seed(row["setup_a_seed"])
        torch.manual_seed(row["setup_a_seed"])
        setup_a_img = apply_and_plot_ready(pil_img, transform_dict["setup_a"], train_mean, train_std)

        random.seed(seed + 2000 + row_idx)
        np.random.seed(seed + 2000 + row_idx)
        torch.manual_seed(seed + 2000 + row_idx)
        setup_b_img = apply_and_plot_ready(pil_img, transform_dict["setup_b"], train_mean, train_std)

        for col_idx, image in enumerate([original_img, base_img, setup_a_img, setup_b_img]):
            ax = axes[row_idx, col_idx]
            ax.imshow(image)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], pad=10)

        axes[row_idx, 0].text(
            -0.08,
            0.5,
            f"Sample {row_idx + 1}",
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
        )

    fig.tight_layout(rect=[0.04, 0.03, 1, 0.93])
    save_figure(fig, figure_dir, "figure_2_preprocessing_augmentation_examples.png")


## 5.3 Figure 4 Experiment 1 training curves

def plot_exp1_training_curves(artifacts: Dict[str, object], figure_dir: Path) -> None:
    series = extract_train_macro_f1_series(
        artifacts["exp1_history"],
        {
            "baseline_cnn": "Baseline CNN",
            "residual_cnn": "Residual CNN",
            "residual_attention_cnn": "Residual Attention CNN",
        },
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    for label in ["Baseline CNN", "Residual CNN", "Residual Attention CNN"]:
        y = np.array(series[label], dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, label=label, color=ARCH_COLORS[label], linewidth=2.2, marker="o", markersize=3.5)

    ax.set_title("Experiment 1 Training Macro-F1 by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Macro-F1")
    ax.set_xticks(np.arange(1, max(len(v) for v in series.values()) + 1))
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_figure(fig, figure_dir, "figure_4_exp1_training_macro_f1_curves.png")


## 5.4 Figure 5 Experiment 2 training curves

def plot_exp2_training_curves(artifacts: Dict[str, object], figure_dir: Path) -> None:
    series = extract_train_macro_f1_series(
        artifacts["exp2_history"],
        {
            "setup_a": "Setup A",
            "setup_b": "Setup B",
            "setup_c": "Setup C",
        },
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    for label in ["Setup A", "Setup B", "Setup C"]:
        y = np.array(series[label], dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, label=label, color=SETUP_COLORS[label], linewidth=2.2, marker="o", markersize=3.5)

    ax.set_title("Experiment 2 Training Macro-F1 by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Macro-F1")
    ax.set_xticks(np.arange(1, max(len(v) for v in series.values()) + 1))
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_figure(fig, figure_dir, "figure_5_exp2_training_macro_f1_curves.png")


## 5.5 Figure 6 final confusion matrix

def plot_final_confusion_matrix(artifacts: Dict[str, object], figure_dir: Path) -> None:
    cm = np.array(artifacts["confusion_matrices"]["experiment_2"]["confusion_matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(cm, cmap=HEATMAP_CMAP)

    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Final Winning Setup: Confusion Matrix")

    max_val = np.max(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > max_val * 0.55 else "black"
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color=text_color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.08)
    cbar.set_label("Count", rotation=270, labelpad=18)

    fig.tight_layout()
    save_figure(fig, figure_dir, "figure_6_final_confusion_matrix.png")


## 5.6 Figure 7 per-class recall

def plot_per_class_recall(artifacts: Dict[str, object], figure_dir: Path) -> None:
    df_recall = tidy_recall_df(artifacts["exp2_final_per_class_recall"])

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bar_colors = [CLASS_COLORS[c] for c in df_recall["class"]]

    bars = ax.bar(
        df_recall["class"].astype(str),
        df_recall["recall"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title("Final Winning Setup: Per-Class Recall")
    ax.set_xlabel("Class")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)

    for bar, val in zip(bars, df_recall["recall"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    save_figure(fig, figure_dir, "figure_7_per_class_recall.png")


## 6.1 Main workflow

def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    set_report_style()
    maybe_mount_drive(args.use_drive)

    paths = get_paths(args.project_name, args.use_drive)
    artifacts = load_plotting_artifacts(paths)

    plot_class_distribution(artifacts, paths["figure_dir"])
    plot_exp1_training_curves(artifacts, paths["figure_dir"])
    plot_exp2_training_curves(artifacts, paths["figure_dir"])
    plot_final_confusion_matrix(artifacts, paths["figure_dir"])
    plot_per_class_recall(artifacts, paths["figure_dir"])

    if not args.skip_image_figure:
        plot_preprocessing_augmentation_examples(artifacts, paths, paths["figure_dir"], seed=args.seed)

    print(f"saved report figures to {paths['figure_dir']}")


## 6.2 CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate final cassava report figures")
    parser.add_argument("--project-name", default=PROJECT_NAME)
    parser.add_argument("--use-drive", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-image-figure", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
