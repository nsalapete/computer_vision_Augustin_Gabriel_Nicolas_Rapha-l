# Emoji provider classification

Computer vision project by Augustin, Gabriel, Nicolas and Raphaël.

Each image is the same kind of emoji drawn by one of **7 providers**: `apple`, `facebook`, `google`, `messenger`, `mozilla`, `samsung`, `whatsapp`. The task, from a Kaggle competition, is to predict the provider of 9,879 unlabelled test images from 9,879 labelled training images, and submit a CSV with columns `Id,Label`.

<!-- TODO: add the Kaggle competition link -->
Competition page: _TODO: link_

Everything is in one notebook, [`emoji_classifier.ipynb`](emoji_classifier.ipynb):

1. PNG decoding with **premultiplied alpha** (removes the hidden colours stored under transparent pixels), padded to a square, resized to 160 px and composited on white
2. **ConvNeXt-Tiny** (ImageNet-22k pretrained, via `timm`), fine-tuned for 12 epochs with AdamW, one-cycle LR, label smoothing and mixed precision, with no data augmentation
3. **Stratified group 5-fold** cross-validation, where exact duplicate images are kept in the same fold
4. Test prediction = average of the 5 fold models, plus a lookup for test images identical to a training image
5. Ablation study (geometric augmentation, horizontal flip, naive RGB conversion) and error analysis

## Results (out-of-fold, 5-fold CV)

| model | OOF accuracy | OOF macro-F1 |
|---|---|---|
| metadata only (decision tree on native size + PNG mode) | 66.70 % | 0.486 |
| **ConvNeXt-Tiny, 5-fold ensemble (final)** | **97.56 %** | **0.976** |

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| accuracy | 97.87 % | 96.61 % | 97.47 % | 97.82 % | 98.03 % |
| macro-F1 | 0.9799 | 0.9678 | 0.9740 | 0.9789 | 0.9805 |

Ablation (fold 0, one change at a time):

| variant | accuracy | macro-F1 |
|---|---|---|
| final recipe | 97.87 % | 0.9799 |
| + geometric augmentation | 95.14 % | 0.9537 |
| + horizontal flip (train + TTA) | 97.47 % | 0.9754 |
| naive RGB conversion | 97.82 % | 0.9776 |

## Repository layout

```
emoji_classifier.ipynb   the whole pipeline: data analysis, training, evaluation, submission
requirements.txt         exact package versions (Python 3.12.10)
train/                   9,879 labelled training PNGs (00001.png ... 09879.png)
train_labels.csv         Id,Label for the training images
test/                    9,879 test PNGs (10001.png ... 19879.png)
submission.csv           test predictions, Id,Label
outputs/                 training history, ablation scores, OOF and test probabilities
AI_USAGE.md              declaration of AI assistance
```

## Setup

You need **Python 3.12**. An NVIDIA GPU is strongly recommended: the notebook takes about 30 minutes on an RTX 4070 Laptop GPU, but well over a day on a laptop CPU.

### Local machine

```bash
git clone https://github.com/nsalapete/computer_vision_Augustin_Gabriel_Nicolas_Rapha-l.git
cd computer_vision_Augustin_Gabriel_Nicolas_Rapha-l

python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# PyTorch first, from the CUDA 12.8 index (no NVIDIA GPU: use https://download.pytorch.org/whl/cpu)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Google Colab

Choose *Runtime → Change runtime type → GPU*, then run in a first cell:

```python
!git clone https://github.com/nsalapete/computer_vision_Augustin_Gabriel_Nicolas_Rapha-l.git
%cd computer_vision_Augustin_Gabriel_Nicolas_Rapha-l
!pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
!pip install -q -r requirements.txt
```

If pip replaced Colab's preinstalled PyTorch, restart the session (*Runtime → Restart session*), run `%cd /content/computer_vision_Augustin_Gabriel_Nicolas_Rapha-l` again, then open `emoji_classifier.ipynb` from the file browser.

## Data

The images ship with the repository: `git clone` downloads `train/`, `test/` and `train_labels.csv` (about 290 MB). Nothing else is needed.

If you only have the code, download the data from the competition's *Data* tab (or with `kaggle competitions download -c <competition-slug>`), unzip it, and either put `train/`, `test/` and `train_labels.csv` next to the notebook or point the notebook to them:

```bash
export EMOJI_DATA_DIR=/path/to/data      # Windows PowerShell: $env:EMOJI_DATA_DIR = "C:\path\to\data"
```

The pretrained weights (about 110 MB) are downloaded from the Hugging Face Hub on the first run, at a fixed commit, so the starting point never changes.

## Running the notebook end to end

From the project folder, open the notebook (`jupyter lab emoji_classifier.ipynb`) and run all cells top to bottom, or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 emoji_classifier.ipynb
```

Approximate runtime on an RTX 4070 Laptop GPU:

| step | time |
|---|---|
| image loading and preprocessing | 2 min |
| 5-fold training | 18 min |
| ablation (section 7, 3 extra training runs) | 11 min |

To skip the ablation, set `RUN_ABLATION = False` in section 7. It only produces `outputs/ablation.csv` and does not affect the submission.

All seeds are fixed and cuDNN runs in deterministic mode, so a rerun on the same GPU and package versions reproduces the same predictions. Results on a different GPU or on the CPU can differ in the last decimals.

## Outputs

| file | content |
|---|---|
| `submission.csv` | 9,879 rows, `Id,Label` (Ids keep their leading zeros, labels are the 7 provider names); the notebook checks the file against the test folder |
| `outputs/history.csv` | loss, accuracy and macro-F1 per epoch and fold |
| `outputs/oof_proba.npy` | out-of-fold class probabilities for the training images, shape (9879, 7) |
| `outputs/test_proba.npy` | 5-fold averaged class probabilities for the test images, shape (9879, 7) |
| `outputs/ablation.csv` | fold-0 scores of the ablation variants |

Class order in the `.npy` files: `apple, facebook, google, messenger, mozilla, samsung, whatsapp`.
