# Applied AI Midterm — SRGAN Super-Resolution for Dogs vs. Cats

**Author:** Shamik Desai

This repository trains a Super-Resolution GAN (SRGAN) that turns 32×32 images of cats and dogs into 128×128 images. It then measures how much class information those generated images carry by comparing three classifiers built with one identical recipe:

| Model | Trained and tested on | Purpose |
|---|---|---|
| **A** | Real 128×128 images | Upper bound (transfer learning, as in Assignment 1) |
| **B** | SRGAN output (32→128) | The exam's question: are SRGAN images as useful as real ones? |
| **C** | Bicubic upsampling (32→128) | Control: does the SRGAN beat plain interpolation? |

Everything runs in one notebook, [`SRGAN_Midterm.ipynb`](SRGAN_Midterm.ipynb), on Google Colab with a GPU. Every stage saves to Google Drive and resumes automatically after a disconnect.

---

## 1. Repository contents

| File | What it is |
|---|---|
| `SRGAN_Midterm.ipynb` | The complete pipeline: data, classifier A, SRGAN, classifiers B and C, evaluation, workbook |
| `README.md` | This document: how to reproduce the results step by step |
| `Midterm_Results.xlsx` | Results workbook, one row per decision (add after your run) |
| `figures/` | Figures exported by the notebook (add after your run) |

## 2. Requirements

* **Google Colab with a GPU** (Runtime → Change runtime type → GPU). Colab Pro is recommended because the SRGAN trains for 170 epochs in total.
* **Google Drive** with about **5 GB** free (see section 8).
* No installs are needed: Colab already provides PyTorch, torchvision, scikit-learn, SciPy, pandas, openpyxl and matplotlib. The notebook prints the PyTorch version it ran with.
* Internet access on first run, to download the ImageNet weights for ResNet50 and VGG19 from torchvision.

## 3. Data setup

The dataset is the Kaggle **Dogs vs. Cats** archive used in Assignment 1.

1. Upload `archive.zip` to Google Drive at:
   `My Drive/6119_datasets/archive.zip`
2. The archive must contain this structure:
   ```
   dogs-vs-cats-classification/
       train/{cats,dogs}/*.jpg        19,943 images
       validation/{cats,dogs}/*.jpg    2,492 images
       test/{cats,dogs}/*.jpg          2,495 images
   ```
   That is 24,930 images in total. Labels follow Assignment 1: **cats = 0, dogs = 1**.

If your archive lives elsewhere, edit `ARCHIVE_PATH` in the config cell.

## 4. How to run

### 4.1 Optional test run (about 15–30 minutes)

1. Open `SRGAN_Midterm.ipynb` in Colab and select a GPU runtime.
2. In the config cell, set `QUICK_RUN = True`.
3. Runtime → Run all, and allow the Google Drive mount when prompted.

This uses 1,200 images, 2 warm-start epochs, 4 GAN epochs and 3 classifier epochs. It writes to a separate folder (`srgan_midterm_quick`), so it never mixes with the real run. It checks that paths, Drive access and every cell work before you commit many GPU hours. **It is not a valid submission run.**

After the test run, check three things before starting the full run:

1. **Classifier A's validation accuracy is well above 0.5** (with ImageNet weights it should already be high after 3 epochs). If it isn't, the data or labels are wrong.
2. **The "Weighted G loss terms" plot** in section 7. The pixel, perceptual and adversarial terms should be within a couple of orders of magnitude of each other. If the perceptual term dwarfs the others, the L7 weights need revisiting before spending GPU hours.
3. **The progress grids** under `srgan_midterm_quick/srgan/progress/` show recognizable animals: blurry is fine at this stage, noise is not.

### 4.2 Full run

1. Set `QUICK_RUN = False` (the default).
2. Runtime → Run all.
3. **When Colab disconnects** (sessions are time-limited): reconnect, select the GPU runtime again, and **Run all** again. Each stage checks Drive and continues where it stopped:

| Stage | Saved to Drive | On re-run |
|---|---|---|
| Image cache | `cache/hr_uint8.npy`, `cache/meta.csv` | Loaded instead of re-decoding |
| Split | `cache/split_indices.json` | Recomputed deterministically (same seed); file kept for reference |
| Classifiers A, B, C | `classifiers/*.pt` | Loaded instead of retrained |
| SRGAN warm start | `srgan/pretrain_latest.pt`, `srgan/pretrain_final_G.pt` | Resumes from the last saved epoch |
| SRGAN GAN training | `srgan/latest.pt`, `srgan/best.pt`, `srgan/snapshots/` | Resumes from the last saved epoch |
| SR and bicubic image sets | `cache/sr_uint8_epochXXX.npy`, `cache/bicubic_uint8.npy` | Loaded instead of regenerated |

At most 5 epochs of SRGAN work are lost to a disconnect, because full checkpoints are written every 5 epochs.

## 5. Pipeline, step by step

Decision IDs in brackets (for example **[D1]**) refer to the design decisions listed in section 6. The same IDs appear as comments in the notebook code.

### Step 1 — Load, pool and cache the data (notebook section 1)

1. Extract `archive.zip` to local Colab storage (`/content/dataset`), exactly as in Assignment 1.
2. Pool all 24,930 images. The archive's 80/10/10 split does not meet the required 70/30, so it is ignored.
3. Resize every photo so its **shorter side is 128 px, then center-crop to 128×128** [D1]. This keeps animal proportions instead of squashing them.
4. Decode each JPEG once and **cache the 128×128 images as a uint8 array on Drive** [D3]. Low-resolution images are computed on the fly from this cache.
5. Count originals whose shorter side is under 128 px. They are **kept and reported** [D6].
6. **Split:** stratified 70% train / 30% test with `random_state=6165`, then 10% of train held out as validation, also stratified with `random_state=6165`. This gives about 15,705 train, 1,746 validation and 7,479 test images, each split 50/50 cats and dogs. The validation set is used for **all** selection (early stopping, SRGAN checkpoint choice). The test set is used once, at the end.

### Step 2 — Scaling, normalization and transformations (notebook section 2)

* **Downscaling 128→32:** bicubic interpolation **with antialiasing** [D2] (`F.interpolate(..., mode='bicubic', antialias=True)`). The same function creates the SRGAN inputs, the test inputs and control C's inputs.
* **SRGAN value ranges:** LR input in [0, 1]; HR targets and generator output in [−1, 1] [D5].
* **SRGAN augmentation:** paired random horizontal flip [D4]. The HR image is flipped first and the LR image is computed from it, so the pair always matches.
* **Classifier normalization:** ImageNet mean/std [K4].
* **Classifier augmentation:** random horizontal flip only [K8].

The notebook displays sample grids of HR, LR, bicubic and flipped images, the ImageNet-normalized images, and pixel histograms before and after normalization.

### Step 3 — Classifier A (notebook section 3)

The recipe follows Assignment 1, ported to PyTorch [K0], with two Assignment 1 bugs fixed:

* Every epoch now uses the full training set. In Assignment 1, `steps_per_epoch` with a non-repeating generator made every other epoch a single batch.
* Inputs now get the backbone's ImageNet preprocessing instead of rescaling to [0, 1] only.

| Setting | Value |
|---|---|
| Backbone | ResNet50, ImageNet weights [K1], fully frozen with BatchNorm in eval mode [K2] |
| Head | Global average pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(1) [K3]; 524,801 trainable parameters, the same as Assignment 1's ResNet50 build |
| Loss | Binary cross-entropy (`BCEWithLogitsLoss`, equivalent to sigmoid + BCE) |
| Optimizer | Adam, learning rate 1e-4, head parameters only [K5] |
| Epochs | Up to 15; early stopping on validation loss with patience 5; best weights restored [K6] |
| Batch size | 64 [K7] |
| Input | 128×128, ImageNet-normalized [K4], random horizontal flips in training [K8] |

The same function trains A, B and C, so the only difference between them is the image source.

### Step 4 — SRGAN architecture (notebook section 4)

**Generator** (1,546,435 parameters):

```
LR 32x32x3 in [0,1]
 -> Conv 9x9, 64 + PReLU                       (f0)
 -> 16 residual blocks [G1]: Conv3x3 -> PReLU [G4] -> Conv3x3, + input
      no BatchNorm [G2]; weights Kaiming x 0.1 [G5]
 -> Conv 3x3, 64, then + f0 (global skip) [G6]
 -> 2 x [Conv 3x3 to 256 -> PixelShuffle(2) -> PReLU]   (ICNR init) [G3]
 -> Conv 9x9, 3 -> tanh [G7]
SR 128x128x3 in [-1,1]
```

**Discriminator** (SRGAN paper design [L1], about 38.2M parameters):

```
128x128x3 in [-1,1]
 -> 8 x [Conv 3x3 -> LeakyReLU(0.2)], channels 64,64,128,128,256,256,512,512, strides 1,2,1,2,1,2,1,2
 -> Flatten (512x8x8) -> Dense 1024 -> LeakyReLU(0.2) -> Dense 1 (logit)
 spectral normalization on every layer, no BatchNorm [L2]
```

**Losses:**

* Generator: L_G = 1 · L1 pixel [L5] + 6×10⁻³ · VGG perceptual + 10⁻³ · adversarial [L7].
* Perceptual loss: MSE between VGG19 features taken **before the conv5_4 activation** (`vgg19.features[:35]`) of the SR and HR images [L6].
* Adversarial loss: vanilla non-saturating GAN loss, i.e. BCE with target "real" for G [L3].
* Discriminator: BCE with **one-sided label smoothing** (real = 0.9, fake = 0) [L4].

### Step 5 — SRGAN training (notebook sections 5–7)

1. **Warm start, 20 epochs** [T1]: L1 pixel loss only. The result is saved as `pretrain_final_G.pt`.
2. **Adversarial training, 150 epochs** [T1], starting from the warm-started generator:
   * Adam, learning rate 1e-4, betas (0.9, 0.999), for both G and D [T2].
   * Learning rates multiplied by 0.1 after GAN epoch 100 [T3].
   * Batch size 32 [T4]; mixed precision with `autocast` and `GradScaler` [T5].
   * One discriminator step, then one generator step, per batch.
3. **Every 5 epochs** [T6]:
   * Save a generator-only snapshot (about 6 MB).
   * Save the full resumable state to `latest.pt`: G, D, optimizers, schedulers, AMP scalers, history and random-number states.
   * Super-resolve the validation set and record PSNR, SSIM and **classifier A's accuracy on the SR images**.
   * Save a progress grid of the same 4 cats and 4 dogs (LR / bicubic / SR / HR).
   * Save `best.pt` when accuracy improves.

The adversarial stage alone is 150 epochs, meeting the exam's minimum; the 20 warm-start epochs come on top.

### Step 6 — Choosing the generator checkpoint (notebook section 7)

The generator used for model B's data is the snapshot with the **highest classifier-A accuracy on SR validation images** [T7]. Snapshots within 1 accuracy point of the best are treated as tied (about 2 standard errors on roughly 1,750 images), and the latest of them is chosen. The rule is fixed before any test-set evaluation.

### Step 7 — Image sets for B and C (notebook section 8)

Every image is downscaled to 32×32 [D2], then:

* **SR set:** upscaled by the chosen generator.
* **Bicubic set:** upscaled by bicubic interpolation.

Both are saved as uint8 arrays, quantized exactly like the real images. PSNR and SSIM on the test split are reported for SRGAN vs. bicubic.

### Step 8 — Models B and C (notebook section 9)

B is trained on the SR training images, and C on the bicubic training images. Both use the classifier recipe from step 3 unchanged, with the same seed.

### Step 9 — Evaluation (notebook section 10)

All results are on the 30% test split (about 7,479 images). Six pairings are scored [K9]:

| Pairing | Question it answers |
|---|---|
| A: real → real | Ceiling with real images |
| B: SRGAN → SRGAN | Class information in SRGAN images |
| B: SRGAN → real | Does training on SR images transfer to real photos? |
| C: bicubic → bicubic | Control without the GAN |
| A: real → SRGAN | How far SR images sit from real ones (diagnostic) |
| A: real → bicubic | Same, for bicubic (diagnostic) |

For each pairing the notebook reports:

* Accuracy, F1 (dog = positive class) and AUC, each with a 95% bootstrap interval.
* Per-class recall.
* Confusion matrices and ROC curves.
* McNemar's exact test for A vs. B, B vs. C and A vs. C (all on the same test images).

### Step 10 — Results workbook (notebook section 11)

The notebook writes `Midterm_Results.xlsx` with one row per design decision and four columns: Task Name, Method Used, Implementation short description and Primary Outcome. Primary outcomes are filled in automatically from the run where the notebook measures them.

## 6. Design decisions

| ID | Decision | Method used |
|---|---|---|
| D1 | Non-square photo → 128×128 | Resize shorter side to 128, center crop |
| D2 | Degradation 128→32 | Bicubic with antialiasing |
| D3 | Data loading | Decode once, cache uint8 HR on Drive, build LR on the fly |
| D4 | SR-pair augmentation | Paired random horizontal flip |
| D5 | Value ranges | LR in [0,1], HR in [−1,1] |
| D6 | Tiny originals | Keep, report the count |
| K0 | Framework | PyTorch |
| K1 | Backbone | ResNet50 |
| K2 | Fine-tuning | Frozen base, train head only |
| K3 | Head | GAP → Dense(256, ReLU) → Dropout(0.5) → Dense(1, sigmoid) |
| K4 | Input preprocessing | ImageNet mean/std |
| K5 | Optimizer | Adam 1e-4, BCE |
| K6 | Epochs / early stopping | Max 15, patience 5, restore best |
| K7 | Batch size | 64 |
| K8 | Classifier augmentation | Horizontal flip only |
| K9 | What B is tested on | Both SR and real test images |
| G1 | Residual blocks | 16 |
| G2 | Normalization in blocks | None |
| G3 | Upsampling | Conv → PixelShuffle(2), ICNR init |
| G4 | Activation | PReLU |
| G5 | Initialization | Kaiming × 0.1 in residual blocks |
| G6 | Long skip | Global skip from first conv to end of trunk |
| G7 | Output | Tanh |
| L1 | Discriminator | SRGAN paper design |
| L2 | D normalization | Spectral norm |
| L3 | Adversarial loss | Vanilla non-saturating BCE |
| L4 | D labels | One-sided smoothing, real = 0.9 |
| L5 | Pixel loss | L1 |
| L6 | Perceptual loss | VGG19 before conv5_4 activation |
| L7 | Loss weights pix / perc / adv | 1 / 6e-3 / 1e-3 |
| T1 | Warm start | 20 pixel-only epochs, then 150 GAN epochs |
| T2 | Optimizers | Adam 1e-4, betas (0.9, 0.999) |
| T3 | LR schedule | ×0.1 at GAN epoch 100 |
| T4 | SRGAN batch size | 32 |
| T5 | Precision | Mixed precision |
| T6 | Checkpoints | Every 5 epochs |
| T7 | Checkpoint for B's data | Best classifier-A accuracy on SR validation images |

## 7. Exam requirements checklist

| Requirement | Where it is met |
|---|---|
| Train classifier A with transfer learning (as in Assignment 1), images at 128×128 | Step 3, notebook section 3 |
| Train the SRGAN to generate 128×128 images from 32×32 | Steps 4–5, notebook sections 4–7 |
| Show examples of scaled images in the notebook | Notebook section 2 (`figures/scaled_examples.png`) and section 8 |
| Use SRGAN images to train model B | Steps 7–8, notebook sections 8–9 |
| Train the SRGAN for at least 150 epochs | 150 adversarial epochs after a 20-epoch warm start (step 5) |
| 70% training / 30% testing | Step 1 |
| Normalization and image transformations, with examples | Step 2, notebook section 2 (`figures/transformed_examples.png`) |
| Compare models with F1, Accuracy and AUC | Step 9, notebook section 10 |
| Save models every n epochs (Colab limits) | Every 5 epochs (step 5, section 4.2 table) |

## 8. Outputs and storage

All outputs go to `My Drive/srgan_midterm/`:

```
srgan_midterm/
  cache/
    hr_uint8.npy                 ~1.2 GB  real 128x128 images (uint8)
    meta.csv                              file paths, labels, original sizes
    split_indices.json                    train / val / test indices
    sr_uint8_epochXXX.npy        ~1.2 GB  SRGAN images from the chosen generator
    bicubic_uint8.npy            ~1.2 GB  bicubic images (control C)
  classifiers/
    A_real_seed6165.pt, B_srgan_epXXX_seed6165.pt, C_bicubic_seed6165.pt   (head weights + history;
                                  B's name includes the chosen generator epoch, so changing it retrains B)
  srgan/
    pretrain_latest.pt, pretrain_final_G.pt
    latest.pt, best.pt           ~0.5 GB each (G, D and both optimizers)
    snapshots/G_epochXXX.pt      ~6 MB each, 30 files
    progress/epoch_XXX.png       fixed progress grids
    gan_log.csv                  per-epoch losses
  figures/                       sample grids, training curves, confusion matrices, ROC curves, test_results.csv
  Midterm_Results.xlsx
```

Total is about 5 GB. Most of a full checkpoint is the discriminator's 33.5M-weight dense layer plus its two Adam buffers, which is why only `latest.pt` and `best.pt` hold the full state.

**Runtime:** the log prints seconds per epoch. Multiply the first GAN epoch's time by 150 to estimate the total, and plan for several Colab sessions on a T4. Faster GPUs (L4, A100) reduce this a lot.

## 9. Reproducibility notes

* **Seed:** `SEED = 6165`, set with `set_seed()` before every model construction and run. Data shuffling is seeded per epoch, so a resumed run sees the same batch order as an uninterrupted one.
* **Split:** fully determined by `random_state=6165`. The indices are also saved to `cache/split_indices.json`.
* **GPU non-determinism:** `cudnn.benchmark` and mixed precision make GPU results close but not bit-identical between runs. Small differences (a few tenths of a point) between reruns are expected. Several classifier seeds can be run by adding them to `CLASSIFIER_SEEDS` (for example `[6165, 6166, 6167]`); the notebook then reports mean ± std.
* **Paths:** change `ARCHIVE_PATH` and `WORK_DIR` in the config cell if your Drive layout differs.

## 10. Results

Fill in after the full run (copy from notebook section 10):

| Pairing | Accuracy | F1 (dog) | AUC |
|---|---|---|---|
| A: real → real | | | |
| B: SRGAN → SRGAN | | | |
| B: SRGAN → real | | | |
| C: bicubic → bicubic | | | |

| Image quality on test split | PSNR (dB) | SSIM |
|---|---|---|
| SRGAN | | |
| Bicubic | | |

Chosen generator epoch (T7): ___

## 11. References

* Ledig et al. 2017, *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network* (SRGAN). arXiv:1609.04802
* Wang et al. 2018, *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*. arXiv:1809.00219
* Lim et al. 2017, *Enhanced Deep Residual Networks for Single Image Super-Resolution* (EDSR). arXiv:1707.02921
* Shi et al. 2016, *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*. arXiv:1609.05158
* Aitken et al. 2017, *Checkerboard artifact free sub-pixel convolution* (ICNR). arXiv:1707.02937
* Miyato et al. 2018, *Spectral Normalization for Generative Adversarial Networks*. arXiv:1802.05957
* Salimans et al. 2016, *Improved Techniques for Training GANs*. arXiv:1606.03498
* Johnson et al. 2016, *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. arXiv:1603.08155
* He et al. 2016, *Deep Residual Learning for Image Recognition* (ResNet). arXiv:1512.03385
* Micikevicius et al. 2018, *Mixed Precision Training*. arXiv:1710.03740
* Wang et al. 2004, *Image Quality Assessment: From Error Visibility to Structural Similarity* (SSIM). IEEE TIP 13(4)
