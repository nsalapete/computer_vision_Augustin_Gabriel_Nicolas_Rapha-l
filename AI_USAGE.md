# AI usage

Declaration of the AI assistance used in this project.

Sources: the git history (commits that carry a `Co-Authored-By: Claude` line) and the Claude Code session transcripts. The Claude Code sessions ran on Nicolas's machine, so they are listed under his name. Gabriel's commits of 18–19 October carry no AI co-author line, so they are listed below as "not declared".

## Team contributions

All four members, Augustin, Gabriel, Nicolas and Raphaël, worked on the project, and we consider it a shared team effort. Not all of the work went through each person's own commits, and we did not keep a record of who did what. The git history therefore does not show how the work was shared, and this file does not attribute the uncommitted work to individual members.

## Tools used

<!-- For each tool: name, version or model, what it was used for, who in the team used it. -->

| tool | version / model | used for | used by |
|---|---|---|---|
| Claude Code (Anthropic), IDE extension | Claude Opus 5.5 (`claude-opus-5-5`) | 30 Sep – 1 Oct 2026: designing, writing and running `emoji_classifier.ipynb` (data analysis, preprocessing, training, cross-validation, error analysis, ablation), an exploratory hyperparameter sweep (input size, epochs, ConvNeXt-Small), the first `submission.csv`, and commit `c5f1c1bd` | Nicolas |
| Claude Code (Anthropic), Claude desktop app | Claude Opus 5.5 (`claude-opus-5-5`) | 26 Oct 2026: writing the LaTeX report (`report/`), committing, rebasing and pushing it (commit `46e65a53`), filling in this file, adding the public Kaggle score to the report | Nicolas |
| _to be completed by each member_ | | any AI tool used by Augustin, Gabriel or Raphaël themselves | |

## AI-assisted files and cells

<!-- List each file or notebook cell (section number and first line) produced or changed with AI help,
     how much of it (generated, edited, reviewed only), and how it was checked. -->

Notebook cells are listed by section with the first line of their code cells. "Generated" means written by Claude Code from the prompts below and executed by it on Nicolas's GPU (RTX 4070 Laptop). The markdown cells of each section were generated in the same way.

| file / cell | extent of AI assistance | how it was checked |
|---|---|---|
| `emoji_classifier.ipynb` §0 Setup: `import os`, `CFG = {` | generated (commit `c5f1c1bd`); `CFG["weights_revision"]` and the `sys.version` fix added later by Gabriel (not declared) | package versions printed and pinned in `requirements.txt`; fixed seeds and deterministic cuDNN: two complete runs gave identical losses and accuracies at every epoch |
| §1 The data: `train_df = pd.read_csv(`, `def image_meta(path):`, `def show_grid(` | generated | assert that every labelled Id has a PNG file; counts per class add up to 9,879 |
| §2 Preprocessing: `def hidden_rgb_stats(`, `def load_premultiplied(`, `fp_groups = `, `def composite_on_white(xb):`, `sgkf = StratifiedGroupKFold(` | generated; `nearest_upscale_max` option added later by Gabriel (not declared) | after compositing, every fully transparent pixel equals 1.0; assert that no image fingerprint is shared between the training and validation parts of a fold; 0 duplicate groups with conflicting labels |
| §2 Near-duplicate check: `def phash_on_white(path):` | added by Gabriel in `3f40dfa6` (not declared) | not executed yet (no saved output) |
| §3 Metadata baseline: `meta_X = pd.get_dummies(` | generated | evaluated on the same folds as the CNN (66.70 % accuracy) |
| §4 Model and training: `def normalize(img, model):`, `oof_proba = np.zeros(` | generated; oversampling option of `train_one_fold` added later by Gabriel (not declared) | 5-fold out-of-fold evaluation (97.56 % accuracy, macro-F1 0.976); learning curves; settings compared in the exploratory sweep (160 vs 224 px, 12 vs 15 epochs, ConvNeXt-Tiny vs Small) |
| §5 Evaluation: `oof_pred = oof_proba.argmax(1)`, `fig, axes = plt.subplots(1, 2,`, `wrong = np.where(`, `usual_size = ` | generated | metrics computed with scikit-learn from the saved out-of-fold probabilities (`outputs/oof_proba.npy`) |
| §6 Submission: `model_labels = np.array(CLASSES)[` | generated; read-back check of `submission.csv` added later by Gabriel (not declared) | asserts on the file format (one row per test image, valid labels); the 352 test images identical to a training image were all already predicted with the training label; public Kaggle score 0.98238 |
| §7 Ablation: `RUN_ABLATION = True` | generated | each variant changes one thing and reuses the fold-0 split and seed |
| §8 Experiments: `RUN_EXPERIMENTS = True` and following cells | added by Gabriel in `ca4b67fa` (not declared) | not executed yet (no saved output) |
| `requirements.txt` (version of `c5f1c1bd`) | generated from the environment the notebook ran in; later rewritten by Gabriel in `853de4dc` (not declared) | the notebook prints the versions it runs with |
| `submission.csv`, `outputs/history.csv`, `outputs/oof_proba.npy`, `outputs/test_proba.npy`, `outputs/ablation.csv` | produced by running the generated notebook | same checks as §4–§7 |
| `report/main.tex`, `report/main.pdf` | generated; numbers taken from the notebook outputs and `outputs/`, some recomputed with code (confusion matrix and top-2 accuracy from `oof_proba.npy`, parameter counts with `timm`) | compiled with `pdflatex` without errors or warnings; every page rendered and inspected; numbers cross-checked with the notebook and `README.md` |
| `report/figures/` | extracted from the notebook's saved outputs; `banner.png` assembled from random training images | inspected in the compiled PDF |
| `AI_USAGE.md` (this content) | generated from the git history and the Claude Code session transcripts | prompts below copied from the transcripts |
| `README.md`, `.gitignore`, `AI_USAGE.md` template | written by Gabriel (not declared) | — |

## Prompts used

<!-- Paste the prompts, in order, with the tool they were sent to and the date. -->

Only messages typed or pasted by Nicolas are listed. Automatic messages inside the sessions (background-task notifications, monitor events) are left out, and long pasted documents are described instead of copied. Prompts 9 to 12 are replies picked from buttons suggested by Claude Code.

### Prompt 1

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 30 September 2026
- **Prompt:** two pasted documents followed by one typed line:

```
[pasted document 1: the web article "Emoji Classifier: Techniques & Applications" (about 15,000 characters of background on emoji classifiers)]
[pasted document 2: the competition rules page "WINTER 2026 – Emoji Classification Challenge" (objective, organisation, data, external data, submissions and evaluation, code and reproducibility)]
Do it in a ipynb
```

### Prompt 2

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 1 October 2026
- **Context:** resumes the session of prompt 1, which had stopped the day before with background jobs (including a data-analysis workflow) still unfinished.
- **Prompt:**

```
continue what you were doing
```

### Prompt 3

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 1 October 2026
- **Context:** asked while the exploratory hyperparameter sweep (horizontal flip, 224 px input, 15 epochs, ConvNeXt-Small) was running; a question about following its progress, not an instruction.
- **Prompt:**

```
Where can i see the progression of the sweep ?
```

### Prompt 4

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 1 October 2026
- **Context:** asked during the first complete run of the notebook; a question about following the training, not an instruction.
- **Prompt:**

```
how do i see the log of the model training ?
```

### Prompt 5

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 1 October 2026
- **Context:** progress question during the second complete run of the notebook (the reproducibility check).
- **Prompt:**

```
where are we on the training ?
```

### Prompt 6

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 1 October 2026
- **Context:** progress question near the end of the second run, during the ablation study.
- **Prompt:**

```
where are we ?
```

### Prompt 7

- **Tool:** Claude Code (IDE extension), Claude Opus 5.5
- **Date:** 1 October 2026
- **Context:** after the second run had finished; this produced commit `c5f1c1bd` (notebook, submission, outputs, requirements).
- **Prompt:**

```
git add ., commit and push
```

### Prompt 8

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
I want you to look on that project and do the report on latex following those guidelines : 

1. Intro/Front Page
2. Summary (abstract), table of contents + model chosen (is it a CNN, is it a VLM, etc...)
3. Preprocessing (data augmentation ?)
4. Architecture of the model
5. (optional: Metrics Chosen)
6. Performance
7. AI and Code
8. (optional Conclusion)
```

### Prompt 9

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
Commit the report folder (main.tex, main.pdf, figures) to the repo.
```

### Prompt 10

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
Merge add-latex-report into main locally (fast-forward), don't push.
```

### Prompt 11

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
Rebase local main onto origin/main (report commit on top of Gabriel's 8 commits), don't push.
```

### Prompt 12

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
Push main to origin and delete the stale local add-latex-report branch.
```

### Prompt 13

- **Tool:** Claude Code (Claude desktop app), Claude Opus 5.5
- **Date:** 26 October 2026
- **Prompt:**

```
Ok for the Kaggle score i only have the public it's 0.98238
Also can you fill the file of AI from Gabriel
```
