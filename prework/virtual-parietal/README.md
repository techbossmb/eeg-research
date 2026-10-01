# Virtual parietal channels from a 2-channel frontal EEG

Can a 2-electrode frontal headset (F3/F4 or Fp1/Fp2) stand in for parietal electrodes (P3/P4), and how well does it predict emotional **arousal**?

> **Prework.** An exploratory study to judge which research directions are viable for the main project.

Paper: [`paper/virtual_parietal_paper.pdf`](paper/virtual_parietal_paper.pdf). Author: Babatunde Ishola.

## Research questions
- **RQ1:** Can F3/F4 (or Fp1/Fp2) predict P3/P4 band power? This is the "virtual parietal channel".
- **RQ2:** Can arousal be predicted from the frontal pair, the virtual P3/P4, or both, compared with a real 4-channel montage?
- **RQ3:** Which of the 30 scalp electrodes carry arousal information, and how well does F3/F4 classify emotion categories?

## Data
**FACED** (Chen et al., 2023, *Scientific Data*, [doi:10.1038/s41597-023-02650-w](https://doi.org/10.1038/s41597-023-02650-w)), CC-BY-4.0, BIDS release on NEMAR ([doi:10.82901/nemar.nm000112](https://doi.org/10.82901/nemar.nm000112)).

- **Participants:** 123 participants watched 28 emotion-eliciting film clips (35–130 s each) while 32-channel EEG was recorded.
- **Labels:** after each clip they rated their own arousal and valence on 0–7 sliders. These self-ratings are the labels, one per clip (a *trial*).
- **Subjects used:** subjects 0–38.
- **Not included:** no raw data is stored in this repo, and no video data is used, only EEG.

## Pipeline
```bash
pip install -r requirements.txt
git clone https://github.com/nemarDatasets/nm000112 data/bids   # BIDS metadata + event files (run inside prework/virtual-parietal/)
bash download_data.sh                 # raw BDF files, subjects 0-38 (~7 GB)
python3 prepare.py                    # 6 channels, 1-45 Hz, 250 Hz, trials -> data/prepared/
python3 train.py --frontal F3F4       # RQ1 + RQ2  (also --frontal Fp1Fp2)
python3 ablation_aux.py ridge F3F4    # RQ1 feature ablation (ridge|mlp, F3F4|Fp1Fp2)
python3 extract_all_channels.py       # RQ3: per-trial band power, all 30 channels -> data/allch/
python3 channel_scan.py               # RQ3: single-channel / pair scan + emotion classification
python3 train_waveform.py             # optional: 1-D CNN raw-waveform reconstruction (needs PyTorch)
```
All scripts are resumable: re-running skips finished steps.

## Method in brief
- **Features:** 4 s windows (2 s step). Per channel: log power in δ/θ/α/β/γ, relative power, Hjorth mobility and complexity, spectral entropy and log variance. Everything is z-scored within each subject, which amounts to a per-user calibration.
- **Models:** ridge regression, and a fully connected neural net (scikit-learn MLP, 256 and 128 ReLU units, Adam, early stopping).
- **Evaluation:** 5-fold cross-validation split by subject, so every score is for people the model never saw.
- **High/low accuracy:** RQ1 and RQ2 are regressions. For "high/low accuracy", predictions and targets are split at each subject's own median (chance is 50%). The dataset has no high/low label; this split is ours.
- **Simulated headsets:** a "headset" is simulated by keeping only its two electrodes from the 32-channel recording.

## Key results (39 subjects)
| | F3/F4 | Fp1/Fp2 |
|---|---|---|
| RQ1 P3/P4 band power, within-subject r (copy baseline / ridge / NN) | 0.38 / 0.44 / 0.43 | 0.26 / 0.32 / 0.31 |
| RQ1 high/low accuracy (copy / ridge / NN) | 62 / 64 / 63 % | 58 / 60 / 60 % |
| RQ2 arousal high/low accuracy: pair only / virtual P3/P4 / real P3/P4 | 53 / 58 / 54 % | 52 / 55 / 54 % |

- **RQ3 channel scan:** the best electrode pairs combine a left fronto-central site with a temporal one (e.g. F7–T5, FC5–T4) and reach 59%. The best score with shuffled labels was 55.7%. P3–P4 ranks 264th of 435 pairs.
- **RQ3 emotion classification:** F3/F4 alone gets 42% balanced accuracy on positive/negative/neutral (chance 33%) and 23% on 9 emotions (chance 11%). All 30 channels get 56% and 42%.
- **Takeaway:** frontal electrodes recover the part of parietal power they share with the frontal sites, but not the parietal-only part (P4−P3 asymmetry r ≈ 0.03). Arousal is hard to decode from EEG at any electrode in this dataset.

## Repo contents
- **Code:** `*.py`, `download_data.sh`
- **`annex_keys.tsv`:** storage keys for the raw files
- **`results/`:** result JSONs and plots
- **`paper/`:** the write-up
