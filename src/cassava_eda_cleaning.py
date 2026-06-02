"""
cassava_eda_cleaning.py

Prepare the Cassava Leaf Disease Classification dataset, run EDA, create fixed
stratified splits, compute train-only normalization statistics, and save reusable
artifacts for the experiment and reporting scripts.
"""

## 0.1 Imports

import argparse
import json
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


## 0.2 Constants

SEED = 42
PROJECT_NAME = "cass_proj_csci5922"
KAGGLE_COMPETITION = "cassava-leaf-disease-classification"
LABEL_ABBREVIATIONS = {
    "Cassava Bacterial Blight (CBB)": "CBB",
    "Cassava Brown Streak Disease (CBSD)": "CBSD",
    "Cassava Green Mottle (CGM)": "CGM",
    "Cassava Mosaic Disease (CMD)": "CMD",
    "Healthy": "Healthy",
}


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


## 0.4 Project paths

def get_paths(project_name: str, use_drive: bool) -> Dict[str, Path]:
    base_runtime_dir = Path("/content") / project_name
    raw_data_dir = base_runtime_dir / "raw_data"
    extracted_data_dir = base_runtime_dir / "extracted_data"

    if use_drive:
        artifact_dir = Path("/content/drive/MyDrive") / project_name / "artifacts"
    else:
        artifact_dir = base_runtime_dir / "artifacts"

    paths = {
        "base_runtime_dir": base_runtime_dir,
        "raw_data_dir": raw_data_dir,
        "extracted_data_dir": extracted_data_dir,
        "artifact_dir": artifact_dir,
        "eda_output_dir": artifact_dir / "notebook1_eda_outputs",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


## 1.1 Optional Google Drive mount

def maybe_mount_drive(use_drive: bool) -> None:
    if not use_drive:
        return

    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except ModuleNotFoundError:
        raise RuntimeError("Google Drive mounting requires a Colab runtime.")


## 1.2 Kaggle credential setup

def install_kaggle_json(kaggle_json: Path | None) -> None:
    if kaggle_json is None:
        return

    if not kaggle_json.exists():
        raise FileNotFoundError(f"kaggle json not found: {kaggle_json}")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    destination = kaggle_dir / "kaggle.json"
    shutil.copy2(kaggle_json, destination)
    os.chmod(destination, 0o600)


## 1.3 Dataset download and extraction

def download_and_extract_dataset(raw_data_dir: Path, extracted_data_dir: Path, force_download: bool = False) -> None:
    train_csv = extracted_data_dir / "train.csv"
    train_images_dir = extracted_data_dir / "train_images"

    if train_csv.exists() and train_images_dir.exists() and not force_download:
        return

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    extracted_data_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(raw_data_dir)],
        check=True,
    )

    zip_path = raw_data_dir / f"{KAGGLE_COMPETITION}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"downloaded archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_data_dir)


## 2.1 Dataset paths

def get_dataset_paths(extracted_data_dir: Path) -> Dict[str, Path]:
    return {
        "train_csv": extracted_data_dir / "train.csv",
        "sample_submission": extracted_data_dir / "sample_submission.csv",
        "label_map": extracted_data_dir / "label_num_to_disease_map.json",
        "train_images_dir": extracted_data_dir / "train_images",
        "test_images_dir": extracted_data_dir / "test_images",
    }


## 2.2 Load metadata

def load_metadata(train_csv: Path, label_map_path: Path, train_images_dir: Path) -> Tuple[pd.DataFrame, Dict[int, str]]:
    train_df = pd.read_csv(train_csv)

    with open(label_map_path, "r") as f:
        raw_label_map = json.load(f)

    label_to_name = {int(k): v for k, v in raw_label_map.items()}

    train_df["image_path"] = train_df["image_id"].apply(lambda image_id: train_images_dir / image_id)
    train_df["class_name"] = train_df["label"].map(label_to_name)
    train_df["file_exists"] = train_df["image_path"].apply(lambda path: Path(path).exists())

    return train_df, label_to_name


## 2.3 Validate metadata

def validate_metadata(train_df: pd.DataFrame, label_to_name: Dict[int, str], sample_size: int, seed: int) -> Dict[str, object]:
    required_columns = {"image_id", "label", "image_path", "class_name", "file_exists"}
    missing_columns = sorted(required_columns.difference(train_df.columns))

    sample_size = min(sample_size, len(train_df))
    sample_paths = train_df["image_path"].sample(sample_size, random_state=seed).tolist() if sample_size else []

    corrupt_paths = []
    for path in sample_paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupt_paths.append(str(path))

    return {
        "row_count": int(len(train_df)),
        "unique_image_ids": int(train_df["image_id"].nunique()),
        "null_image_ids": int(train_df["image_id"].isnull().sum()),
        "null_labels": int(train_df["label"].isnull().sum()),
        "null_class_names": int(train_df["class_name"].isnull().sum()),
        "missing_files": int((~train_df["file_exists"]).sum()),
        "class_count": int(len(label_to_name)),
        "missing_required_columns": missing_columns,
        "sample_size_checked": int(sample_size),
        "sample_corrupt_files": corrupt_paths,
    }


## 3.1 Clean dataset

def build_clean_dataframe(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    initial_row_count = len(train_df)

    clean_df = train_df.dropna(subset=["image_id", "label"]).copy()
    clean_df = clean_df.loc[clean_df["file_exists"]].copy()
    clean_df = clean_df.reset_index(drop=True)

    summary = {
        "initial_row_count": int(initial_row_count),
        "final_row_count": int(len(clean_df)),
        "rows_removed": int(initial_row_count - len(clean_df)),
    }

    return clean_df, summary


## 4.1 Class distribution table

def build_class_distribution(clean_df: pd.DataFrame, label_to_name: Dict[int, str]) -> pd.DataFrame:
    class_counts = clean_df["label"].value_counts().sort_index()
    class_proportions = clean_df["label"].value_counts(normalize=True).sort_index()

    class_distribution_df = pd.DataFrame({
        "label": class_counts.index,
        "class_name": [label_to_name[i] for i in class_counts.index],
        "class_abbreviation": [LABEL_ABBREVIATIONS.get(label_to_name[i], label_to_name[i]) for i in class_counts.index],
        "count": class_counts.values,
        "proportion": class_proportions.values,
    })

    return class_distribution_df


## 4.2 Image size summary

def add_image_size_columns(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    widths = []
    heights = []

    for path in clean_df["image_path"]:
        with Image.open(path) as img:
            width, height = img.size
        widths.append(width)
        heights.append(height)

    clean_df = clean_df.copy()
    clean_df["width"] = widths
    clean_df["height"] = heights
    clean_df["aspect_ratio"] = clean_df["width"] / clean_df["height"]

    size_summary = {
        "min_width": float(clean_df["width"].min()),
        "max_width": float(clean_df["width"].max()),
        "median_width": float(clean_df["width"].median()),
        "min_height": float(clean_df["height"].min()),
        "max_height": float(clean_df["height"].max()),
        "median_height": float(clean_df["height"].median()),
        "min_aspect_ratio": float(clean_df["aspect_ratio"].min()),
        "max_aspect_ratio": float(clean_df["aspect_ratio"].max()),
        "median_aspect_ratio": float(clean_df["aspect_ratio"].median()),
    }

    return clean_df, size_summary


## 4.3 EDA plots

def save_class_distribution_plot(class_distribution_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "eda_class_distribution.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_distribution_df["class_name"], class_distribution_df["count"])
    ax.set_title("class distribution in cassava training data")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_image_size_plots(clean_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    outputs = {}

    for column, title, filename in [
        ("width", "image width distribution", "eda_width_distribution.png"),
        ("height", "image height distribution", "eda_height_distribution.png"),
        ("aspect_ratio", "image aspect ratio distribution", "eda_aspect_ratio_distribution.png"),
    ]:
        output_path = output_dir / filename
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(clean_df[column], bins=30)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs[column] = output_path

    return outputs


def save_class_examples(clean_df: pd.DataFrame, label_to_name: Dict[int, str], output_dir: Path, samples_per_class: int = 4) -> Path:
    output_path = output_dir / "eda_class_examples.png"
    labels = sorted(clean_df["label"].unique())

    fig, axes = plt.subplots(
        len(labels),
        samples_per_class,
        figsize=(4 * samples_per_class, 4 * len(labels)),
    )

    if len(labels) == 1:
        axes = np.array([axes])

    for row_idx, label in enumerate(labels):
        class_df = clean_df[clean_df["label"] == label].sample(samples_per_class, random_state=SEED)
        for col_idx, (_, row) in enumerate(class_df.iterrows()):
            ax = axes[row_idx, col_idx]
            with Image.open(row["image_path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(label_to_name[int(label)], loc="left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


## 5.1 Stratified splits

def create_stratified_splits(clean_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_split_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        stratify=clean_df["label"],
        random_state=seed,
    )

    val_split_df, test_split_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_split_df = train_split_df.reset_index(drop=True)
    val_split_df = val_split_df.reset_index(drop=True)
    test_split_df = test_split_df.reset_index(drop=True)

    train_split_df["split"] = "train"
    val_split_df["split"] = "validation"
    test_split_df["split"] = "test"

    return train_split_df, val_split_df, test_split_df


def build_split_summary(*splits: Tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, split_df in splits:
        counts = split_df["label"].value_counts().sort_index()
        for label, count in counts.items():
            rows.append({
                "split": split_name,
                "label": int(label),
                "count": int(count),
                "proportion_within_split": float(count / len(split_df)),
            })

    return pd.DataFrame(rows)


## 6.1 Preprocessing decisions

def build_preprocessing_decisions(clean_df: pd.DataFrame) -> Dict[str, object]:
    uniform_shape = clean_df["width"].nunique() == 1 and clean_df["height"].nunique() == 1

    return {
        "final_input_size": 224,
        "final_resize_policy": "direct_resize_square",
        "normalization_source": "train_split_dataset_specific",
        "raw_images_have_uniform_shape": bool(uniform_shape),
        "raw_width": int(clean_df["width"].iloc[0]),
        "raw_height": int(clean_df["height"].iloc[0]),
        "class_imbalance_observed": True,
        "experiment_2_setup_c_imbalance_aware_loss_justified": True,
    }


## 6.2 Train normalization statistics

class CassavaStatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")
        return self.transform(image)


def compute_train_mean_std(train_split_df: pd.DataFrame, image_size: int, batch_size: int, num_workers: int) -> Tuple[list[float], list[float]]:
    dataset = CassavaStatDataset(train_split_df, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    pixel_count = 0

    for images in loader:
        batch_size_current, channels, height, width = images.shape
        pixels = batch_size_current * height * width
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        pixel_count += pixels

    mean = channel_sum / pixel_count
    variance = (channel_squared_sum / pixel_count) - (mean ** 2)
    std = torch.sqrt(variance)

    return mean.tolist(), std.tolist()


## 7.1 Artifact saving

def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "image_path" in out.columns:
        out["image_path"] = out["image_path"].astype(str)
    return out


def save_artifacts(
    artifact_dir: Path,
    clean_df: pd.DataFrame,
    train_split_df: pd.DataFrame,
    val_split_df: pd.DataFrame,
    test_split_df: pd.DataFrame,
    label_to_name: Dict[int, str],
    class_distribution_df: pd.DataFrame,
    split_summary_df: pd.DataFrame,
    preprocessing_config: Dict[str, object],
    validation_summary: Dict[str, object],
    cleaning_summary: Dict[str, object],
    size_summary: Dict[str, float],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataframe_for_csv(clean_df).to_csv(artifact_dir / "clean_df.csv", index=False)
    dataframe_for_csv(train_split_df).to_csv(artifact_dir / "train_split_df.csv", index=False)
    dataframe_for_csv(val_split_df).to_csv(artifact_dir / "val_split_df.csv", index=False)
    dataframe_for_csv(test_split_df).to_csv(artifact_dir / "test_split_df.csv", index=False)
    class_distribution_df.to_csv(artifact_dir / "class_count_summary.csv", index=False)
    class_distribution_df.to_csv(artifact_dir / "dataset_class_distribution.csv", index=False)
    split_summary_df.to_csv(artifact_dir / "split_summary.csv", index=False)

    label_mapping_payload = {
        "label_to_name": {str(k): v for k, v in label_to_name.items()},
        "name_to_label": {v: int(k) for k, v in label_to_name.items()},
    }

    with open(artifact_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping_payload, f, indent=2)

    with open(artifact_dir / "preprocessing_config.json", "w") as f:
        json.dump(preprocessing_config, f, indent=2)

    pipeline_decisions = {
        "validation_summary": validation_summary,
        "cleaning_summary": cleaning_summary,
        "image_size_summary": size_summary,
        "split_policy": "70/15/15 stratified by label",
        "preprocessing_config": preprocessing_config,
    }

    with open(artifact_dir / "pipeline_decisions.json", "w") as f:
        json.dump(pipeline_decisions, f, indent=2)


## 8.1 Main workflow

def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    maybe_mount_drive(args.use_drive)
    install_kaggle_json(Path(args.kaggle_json) if args.kaggle_json else None)

    paths = get_paths(args.project_name, args.use_drive)

    if not args.skip_download:
        download_and_extract_dataset(
            raw_data_dir=paths["raw_data_dir"],
            extracted_data_dir=paths["extracted_data_dir"],
            force_download=args.force_download,
        )

    dataset_paths = get_dataset_paths(paths["extracted_data_dir"])
    train_df, label_to_name = load_metadata(
        train_csv=dataset_paths["train_csv"],
        label_map_path=dataset_paths["label_map"],
        train_images_dir=dataset_paths["train_images_dir"],
    )

    validation_summary = validate_metadata(
        train_df=train_df,
        label_to_name=label_to_name,
        sample_size=args.sample_check_size,
        seed=args.seed,
    )

    clean_df, cleaning_summary = build_clean_dataframe(train_df)
    clean_df, size_summary = add_image_size_columns(clean_df)

    class_distribution_df = build_class_distribution(clean_df, label_to_name)
    train_split_df, val_split_df, test_split_df = create_stratified_splits(clean_df, seed=args.seed)

    split_summary_df = build_split_summary(
        ("train", train_split_df),
        ("validation", val_split_df),
        ("test", test_split_df),
    )

    preprocessing_config = build_preprocessing_decisions(clean_df)
    train_mean, train_std = compute_train_mean_std(
        train_split_df=train_split_df,
        image_size=int(preprocessing_config["final_input_size"]),
        batch_size=args.stat_batch_size,
        num_workers=args.num_workers,
    )

    preprocessing_config.update({
        "final_normalization": "dataset_specific_train",
        "train_mean": train_mean,
        "train_std": train_std,
    })

    save_class_distribution_plot(class_distribution_df, paths["eda_output_dir"])
    save_image_size_plots(clean_df, paths["eda_output_dir"])
    save_class_examples(clean_df, label_to_name, paths["eda_output_dir"], samples_per_class=args.samples_per_class)

    save_artifacts(
        artifact_dir=paths["artifact_dir"],
        clean_df=clean_df,
        train_split_df=train_split_df,
        val_split_df=val_split_df,
        test_split_df=test_split_df,
        label_to_name=label_to_name,
        class_distribution_df=class_distribution_df,
        split_summary_df=split_summary_df,
        preprocessing_config=preprocessing_config,
        validation_summary=validation_summary,
        cleaning_summary=cleaning_summary,
        size_summary=size_summary,
    )

    print(f"saved artifacts to {paths['artifact_dir']}")


## 8.2 CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="prepare cassava data, EDA, splits, and artifacts")
    parser.add_argument("--project-name", default=PROJECT_NAME)
    parser.add_argument("--use-drive", action="store_true")
    parser.add_argument("--kaggle-json", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-check-size", type=int, default=100)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--stat-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
