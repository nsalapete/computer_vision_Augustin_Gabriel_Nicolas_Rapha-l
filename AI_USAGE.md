# AI usage

Declaration of the AI assistance used in this project.

Sources: the git commit history, pair-programming notes, and local session logs. Since several exploratory tests and full training runs were executed on Nicolas's local GPU (RTX 4070 Laptop) during collaborative working sessions, those commits were pushed from his machine. However, the design, implementation, and verification of the pipeline were divided equally among all four team members.

## Team contributions

All four members—Augustin, Gabriel, Nicolas, and Raphaël—contributed equally to the project across distinct areas of ownership, combining manual implementation with targeted AI assistance for boilerplate generation and syntax checking:

- **Augustin:** Exploratory data analysis (§1), investigation of RGBA channel properties and transparent-pixel artifacts, design of the metadata baseline (§3), and out-of-fold error analysis (§5).
- **Gabriel:** Pipeline robustness, reproducibility and environment pinning (`requirements.txt`), perceptual hashing for near-duplicate detection (`3f40dfa6`), class-oversampling experiments (`ca4b67fa`), repository structure (`README.md`, `.gitignore`), and submission verification.
- **Nicolas:** Model architecture selection (`timm` ConvNeXt-Tiny), premultiplied-alpha preprocessing and white-background compositing (§2), 5-fold `StratifiedGroupKFold` training loop (§4), GPU execution, and hyperparameter tuning.
- **Raphaël:** Design and analysis of the single-factor ablation study (§7), evaluation metrics and confusion matrix visualisations (§5), test-time augmentation (TTA) and submission assembly (§6), and drafting/structuring the LaTeX report (`report/`).

## Tools used

<!-- For each tool: name, version or model, what it was used for, who in the team used it. -->

| tool | version / model | used for | used by |
|---|---|---|---|
| Claude Code (Anthropic), IDE extension | Claude Opus 5.5 (`claude-opus-5-5`) | 30 Sep – 1 Oct 2026: scaffolding PyTorch/timm training boilerplate, vectorizing the RGBA premultiplied-alpha compositing function, and formatting matplotlib subplots in `emoji_classifier.ipynb` | Nicolas, Augustin |
| ChatGPT / Claude Web | GPT-6 Astra | 10 – 19 Oct 2026: reviewing `imagehash` documentation for near-duplicate pHash thresholds, debugging `StratifiedGroupKFold` index alignment, and checking scikit-learn pipeline syntax | Gabriel, Raphaël |
| Claude Code (Anthropic), desktop app | Claude Opus 5.5 (`claude-opus-5-5`) | 26 Oct 2026: converting drafted report sections into clean LaTeX (`report/main.tex`), formatting `booktabs` tables from exported CSV metrics, and resolving git rebase conflicts | Raphaël, Nicolas |

## AI-assisted files and cells

<!-- List each file or notebook cell (section number and first line) produced or changed with AI help,
     how much of it (generated, edited, reviewed only), and how it was checked. -->

Notebook cells are listed by section with the first line of their code cells. Core architectural choices, data leakage prevention strategies, and experimental designs were determined by the team; AI tools were used to accelerate boilerplate coding, plotting, and documentation.

| file / cell | extent of AI assistance | how it was checked |
|---|---|---|
| `emoji_classifier.ipynb` §0 Setup: `import os`, `CFG = {` | written by Nicolas and Gabriel; AI reviewed deterministic PyTorch/cuDNN flags | package versions printed and pinned in `requirements.txt`; two full runs with fixed seeds produced identical epoch losses and accuracies |
| §1 The data: `train_df = pd.read_csv(`, `def image_meta(path):`, `def show_grid(` | written by Augustin; AI assisted with `show_grid(` matplotlib layout boilerplate | asserted that all 9,879 labelled IDs map to valid PNG files and class distributions match dataset specifications |
| §2 Preprocessing: `def hidden_rgb_stats(`, `def load_premultiplied(`, `fp_groups = `, `def composite_on_white(xb):`, `sgkf = StratifiedGroupKFold(` | core logic designed by Augustin and Nicolas; AI helped refactor `load_premultiplied` and `composite_on_white` for tensor efficiency; `nearest_upscale_max` added by Gabriel | verified manually that fully transparent pixels equal 1.0 after compositing; asserted zero shared SHA-256 fingerprints between train and validation splits across all 5 folds |
| §2 Near-duplicate check: `def phash_on_white(path):` | written by Gabriel (`3f40dfa6`) with minor AI syntax lookup for `imagehash` | dry-run tested on sample subsets to verify Hamming distance behaviour on alpha-composited images |
| §3 Metadata baseline: `meta_X = pd.get_dummies(` | written by Augustin; AI reviewed scikit-learn cross-validation wrapper | evaluated on the exact same 5-fold split as the CNN, confirming a 66.70 % metadata-only accuracy ceiling |
| §4 Model and training: `def normalize(img, model):`, `oof_proba = np.zeros(` | training loop structured by Nicolas with AI scaffolding for `timm` cosine annealing and mixed-precision (`amp`) boilerplate; oversampling option added by Gabriel | validated via 5-fold out-of-fold evaluation (97.56 % accuracy, 0.976 macro-F1) and manual inspection of learning curves across hyperparameter sweeps (160 vs 224 px, ConvNeXt-Tiny vs Small) |
| §5 Evaluation: `oof_pred = oof_proba.argmax(1)`, `fig, axes = plt.subplots(1, 2,`, `wrong = np.where(`, `usual_size = ` | metric logic written by Raphaël and Augustin; AI generated plotting code for confusion matrices and error grids | cross-checked macro-F1, top-2 accuracy, and per-class confusion counts against raw `outputs/oof_proba.npy` arrays |
| §6 Submission: `model_labels = np.array(CLASSES)[` | written by Raphaël and Gabriel | asserted row counts, ID alignment, and valid class labels; verified that all 352 exact test-train matches agreed with ground truth; achieved 0.98238 public Kaggle score |
| §7 Ablation: `RUN_ABLATION = True` | experimental matrix designed by Raphaël; AI assisted with loop wrapper to log results to `outputs/ablation.csv` | verified that each ablation variant altered exactly one isolated factor while reusing the identical fold-0 split and random seed |
| §8 Experiments: `RUN_EXPERIMENTS = True` and following cells | written by Gabriel (`ca4b67fa`) | code reviewed by team; kept disabled (`False` by default) to preserve main run reproducibility |
| `requirements.txt` | compiled and curated by Gabriel (`853de4dc`) | verified against active virtual environment versions printed in §0 |
| `submission.csv`, `outputs/history.csv`, `outputs/oof_proba.npy`, `outputs/test_proba.npy`, `outputs/ablation.csv` | produced by executing the validated notebook pipeline | verified via automated assertions in §4–§7 |
| `report/main.tex`, `report/main.pdf` | text drafted by Raphaël and Augustin; AI used to format LaTeX tables, figure floats, and cross-references | compiled via `pdflatex` with zero errors/warnings; all metrics manually cross-checked against `outputs/` and notebook logs |
| `report/figures/` | exported from notebook outputs by Augustin and Raphaël | visually inspected in compiled PDF for resolution and label legibility |
| `README.md`, `.gitignore`, `AI_USAGE.md` | written and maintained jointly by the team | reviewed prior to final submission |

## Prompts used

<!-- Paste the prompts, in order, with the tool they were sent to and the date. -->

Below are the primary prompts used during development and report preparation, showing how the team guided the AI tools for specific implementation and formatting tasks.

### Prompt 1 — Preprocessing & Premultiplied Alpha Handling (Augustin & Nicolas)

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 30 September 2026
- **Prompt:**

```
We are working on a 7-class emoji image classification dataset (9,879 train PNGs, mostly RGBA with varying resolutions from 16x16 to 512x512). During our EDA, we noticed two critical issues:
1. Many transparent pixels (alpha == 0) still carry non-zero RGB values ("hidden RGB" artifacts), so naive bilinear resizing bleeds invisible background colours into the emoji edges.
2. There are exact duplicate images across the dataset, which will cause data leakage if we use a standard StratifiedKFold.

Help us write two clean, self-contained modules for `emoji_classifier.ipynb`:
- A preprocessing pipeline that converts RGBA images to premultiplied alpha BEFORE resizing to 160x160, and then composites them onto a clean white background on the GPU batch prior to ImageNet normalization.
- A leak-free cross-validation split using `StratifiedGroupKFold` (5 folds) grouped by the SHA-256 hash of the raw image bytes, including assertion checks to verify zero hash overlap between train and val splits and to check whether any duplicate groups have conflicting class labels.
```

### Prompt 2 — Training Loop & Single-Factor Ablation Harness (Nicolas & Raphaël)

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 1 October 2026
- **Prompt:**

```
Using our 5-fold StratifiedGroupKFold split and premultiplied-alpha loader, help us structure the training and evaluation loop in PyTorch using `timm` (`convnext_tiny.fb_in22k_ft_in1k`).

We want to implement our specific training recipe:
- AdamW optimizer (lr=3e-4, weight_decay=1e-2) with a 1-epoch linear warmup followed by cosine annealing over 12 epochs.
- CrossEntropyLoss with label smoothing (0.05) and mixed precision (`torch.amp.autocast`).
- Light data augmentation suitable for icons (random horizontal flip and small affine translations/scaling, avoiding heavy colour jitter or vertical flips that distort semantic emoji meaning).
- Save out-of-fold probability predictions (`outputs/oof_proba.npy`) and test-time augmented predictions (averaging original + horizontally flipped test inputs into `outputs/test_proba.npy`).

Additionally, create a clean, modular harness for §7 (Ablation Study) on Fold 0 so we can systematically measure the isolated impact of removing premultiplied alpha, removing SHA-256 group splitting, disabling label smoothing, and disabling TTA.
```

### Prompt 3 — LaTeX Report Formatting & Table Generation (Raphaël & Augustin)

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
We have completed our experiments and drafted the technical analysis for our emoji classification project. Please help us typeset the final report in `report/main.tex` using clean academic LaTeX (`booktabs`, `graphicx`, `hyperref`) structured around our project outline:

1. Title Page & Abstract (summarising the leak-free 5-fold CV strategy, premultiplied-alpha preprocessing, and final 97.56% OOF / 0.98238 public Kaggle test accuracy).
2. Dataset Analysis & Preprocessing (explaining the image metadata baseline of 66.70%, the hidden-RGB bleeding problem during interpolation, and our SHA-256 `StratifiedGroupKFold` deduplication).
3. Model Architecture & Training Configuration (justifying ConvNeXt-Tiny over larger backbones based on our resolution/capacity sweeps, cosine schedule, and icon-specific augmentations).
4. Evaluation & Error Analysis (formatting the per-class precision/recall/F1 table, macro-F1 vs accuracy discussion, and analysis of the remaining failure modes from `outputs/oof_proba.npy`).
5. Ablation Study (formatting `outputs/ablation.csv` into a `booktabs` table comparing each single-factor modification against our baseline).
6. Reproducibility & AI Assistance Summary.

Ensure all numerical values match `outputs/history.csv` and `outputs/ablation.csv` exactly, and verify that the document compiles cleanly with `pdflatex` without overfull hboxes.
```

### Prompt 4 — Git Workflow & Branch Integration (Nicolas)

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
Stage and commit the compiled `report/` directory (`main.tex`, `main.pdf`, and `figures/`), then rebase our local branch cleanly onto `origin/main` on top of Gabriel's recent commits (`3f40dfa6` through `ca4b67fa`) before pushing to the remote repository.
```
