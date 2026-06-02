# Cassava Lightweight CNN Study

Lightweight and reproducible cassava leaf disease classification using controlled CNN architecture and training-strategy ablations on field images.

## 1. Study explanation

### Project objective

This project studies cassava leaf disease classification as a lightweight, reproducible computer vision problem rather than as a maximum-complexity benchmark competition. The goal is to understand which modeling choices improve performance enough to justify their added complexity.

The study uses the Kaggle Cassava Leaf Disease Classification dataset and frames the task as supervised 5-class image classification. The five target classes are:

- Cassava Bacterial Blight (CBB)
- Cassava Brown Streak Disease (CBSD)
- Cassava Green Mottle (CGM)
- Cassava Mosaic Disease (CMD)
- Healthy

The central question is:

> Which lightweight CNN architecture and training strategy provide the clearest tradeoff among predictive performance, class-sensitive behavior, reproducibility, and practical field-image deployment relevance?

### Motivation

Cassava is a major food-security crop, especially in Africa. Cassava disease detection is therefore not only an image-classification problem; it is connected to crop monitoring, farmer decision-making, and practical agricultural support. Field-image classification is difficult because images can vary in lighting, background, camera distance, orientation, blur, leaf occlusion, and disease visibility.

Because the intended use case is field-image recognition, this study emphasizes:

- lightweight model design
- reproducible experimental setup
- controlled comparisons
- validation macro F1 under class imbalance
- class-level recall and confusion-matrix behavior
- performance-complexity tradeoffs

### Dataset

The project uses the Kaggle Cassava Leaf Disease Classification dataset. The labeled dataset contains 21,397 validated field images.

Key dataset properties:

| Property | Value |
|---|---:|
| Labeled images | 21,397 |
| Image size | 800 x 600 RGB |
| Number of classes | 5 |
| Split strategy | fixed stratified 70/15/15 |
| Training images | 14,977 |
| Validation images | 3,210 |
| Test images | 3,210 |
| Primary selection metric | validation macro F1 |

The class distribution is imbalanced, with CMD as the dominant class. This motivated validation macro F1 as the primary model-selection metric and also motivated testing weighted cross-entropy in the second experiment.

### Preprocessing

The base preprocessing pipeline was fixed before modeling:

- resize images to 224 x 224
- convert images to tensors
- normalize using training-split statistics only

Training-split normalization statistics:

```text
mean = [0.4312, 0.4977, 0.3146]
std  = [0.2203, 0.2234, 0.2116]
```

The raw images all share the same 800 x 600 shape and 4:3 aspect ratio. This simplified preprocessing because the pipeline did not need to handle mixed raw image dimensions.

### Experimental design

The study uses two sequential controlled experiments.

#### Experiment 1: Architecture ablation

Experiment 1 compares three related lightweight CNN architectures while holding the rest of the training pipeline fixed.

| Model | Description |
|---|---|
| Baseline CNN | plain convolutional feature extractor with adaptive average pooling and a linear classifier |
| Residual CNN | adds residual blocks and skip connections |
| Residual Attention CNN | adds lightweight channel attention inside the residual design |

The point of this experiment is to isolate the marginal effect of architectural complexity.

#### Experiment 2: Training-strategy ablation

Experiment 2 fixes the best architecture from Experiment 1, then compares three training strategies.

| Setup | Training strategy |
|---|---|
| Setup A | basic augmentation with random horizontal flipping |
| Setup B | stronger field-oriented augmentation with flipping, rotation, color jitter, and Gaussian blur |
| Setup C | Setup B plus weighted cross-entropy loss |

The point of this experiment is to test whether additional training complexity improves generalization after the model architecture has already been selected.

### Main results

#### Experiment 1 validation results

| Model | Parameters | Best epoch | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|---:|
| Baseline CNN | 390,181 | 28 | 0.8234 | 0.6776 |
| Residual CNN | 1,227,685 | 30 | 0.8399 | 0.7107 |
| Residual Attention CNN | 1,239,981 | 20 | 0.8427 | 0.7260 |

The Residual Attention CNN won the architecture ablation. Residual learning produced the largest absolute gain over the baseline, while lightweight channel attention added only about 1% more parameters beyond the residual model and still improved validation macro F1.

#### Experiment 2 validation results

| Setup | Augmentation | Loss | Best epoch | Validation accuracy | Validation macro F1 |
|---|---|---|---:|---:|---:|
| Setup A | random horizontal flip | cross-entropy | 22 | 0.8617 | 0.7449 |
| Setup B | flip, rotation, color jitter, Gaussian blur | cross-entropy | 30 | 0.8564 | 0.7345 |
| Setup C | same as Setup B | weighted cross-entropy | 24 | 0.8349 | 0.7333 |

Setup A won the training-strategy ablation. In this controlled setting, the simplest augmentation strategy generalized best.

#### Final selected system

The final selected configuration was:

```text
Residual Attention CNN + Setup A
```

Held-out test performance:

| Metric | Value |
|---|---:|
| Test accuracy | 0.8589 |
| Test macro F1 | 0.7365 |
| Test loss | 0.4183 |

Per-class test recall:

| Class | Recall |
|---|---:|
| CBB | 0.509 |
| CBSD | 0.729 |
| CGM | 0.732 |
| CMD | 0.970 |
| Healthy | 0.668 |

The model performed strongest on CMD and weakest on CBB. The confusion-matrix analysis showed that some CBB cases were confused with Healthy leaves, so strong aggregate metrics did not imply equally reliable recognition across all classes.

### Main conclusions

1. Controlled lightweight studies are useful because they make it easier to isolate which design choices are responsible for performance changes.
2. Architectural complexity helped selectively: residual learning improved over the baseline, and lightweight channel attention improved over the residual model at low marginal parameter cost.
3. Training complexity did not help in this setting: stronger augmentation and weighted loss did not outperform the simpler Setup A.
4. Class-level behavior remains important for deployment. The final model achieved strong overall performance but remained uneven across disease classes.

### Limitations

This project is a controlled study, not a production diagnostic system.

Important limitations:

- The final model was evaluated on an internal held-out split, not an external dataset.
- The study used a small family of custom CNNs rather than comparing all possible lightweight architectures or pretrained models.
- The experiments were single-run comparisons, so repeated-seed validation would be needed for stronger statistical confidence.
- CBB recall remained low compared with CMD recall.
- Parameter count was used as the main lightweight-efficiency measure; mobile latency, memory use, and real on-device performance were not benchmarked.

### Future work

Possible extensions include:

- testing on external cassava or plant-disease datasets
- measuring inference latency and memory use on mobile or edge devices
- running repeated-seed experiments
- testing additional lightweight models or transfer-learning baselines
- performing targeted error analysis for CBB vs. Healthy confusion
- adding calibration and uncertainty estimation for safer decision support

## 2. Running the code

### Repository structure

```text
cassava-lightweight-cnn-study/
├── README.md
├── requirements.txt
├── src/
│   ├── cassava_eda_cleaning.py
│   ├── cassava_experiments.py
│   └── cassava_reporting_plots.py
├── reports/
├── presentation/
├── figures/
├── notebooks/
├── data/
└── artifacts/
```

### Environment setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Kaggle setup

This project uses the Kaggle Cassava Leaf Disease Classification competition dataset.

Download your Kaggle API token from your Kaggle account and place `kaggle.json` at:

```text
~/.kaggle/kaggle.json
```

Then set the correct permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Do not commit `kaggle.json` to GitHub.

### Run data preparation and EDA

```bash
python src/cassava_eda_cleaning.py
```

This script:

- downloads and extracts the Kaggle dataset
- validates the metadata and image files
- attaches class names
- checks image dimensions and class distribution
- creates fixed stratified train/validation/test splits
- computes training-only normalization statistics
- saves reusable artifacts for modeling

### Run experiments

```bash
python src/cassava_experiments.py
```

This script:

- loads the saved split and preprocessing artifacts
- rebuilds runtime image paths
- defines the dataset and dataloaders
- trains the Baseline CNN, Residual CNN, and Residual Attention CNN
- runs the training-strategy ablation
- selects best models using validation macro F1
- evaluates selected models on the held-out test set
- saves result tables, histories, metrics, checkpoints, and confusion matrices

### Generate report figures

```bash
python src/cassava_reporting_plots.py
```

This script:

- loads saved experiment artifacts
- generates report-ready figures
- creates class-distribution, learning-curve, confusion-matrix, per-class recall, and augmentation figures
- saves figure outputs for the written report and presentation

### Notes on data and artifacts

The raw dataset, extracted images, model checkpoints, and generated artifacts are intentionally excluded from version control. They are either too large for GitHub or reproducible from the code.

The repository should track:

- source code
- README
- requirements
- report PDF / LaTeX source
- presentation or poster
- selected final figures if they are small enough and useful for review

The repository should not track:

- raw Kaggle images
- extracted image folders
- Kaggle credentials
- large checkpoints
- temporary logs
- generated cache files
