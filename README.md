
# Relay Signal Classifier

**One-line:** Real-time relay fault / attack detection using time- and frequency-domain features and tree-based classifiers (RandomForest / XGBoost).  
**Author:** Sai Kumar (GitHub: saikumardurgavajula)  
**Stack:** Python, Jupyter Notebook, scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn, Streamlit

---

## Project Overview
This repository contains code and models for detecting relay faults/attacks from high-frequency sensor streams (voltage, current, temperature). It includes notebooks for per-relay modeling (Relay1..Relay4), an overall model that aggregates relay outputs, trained model objects and a Streamlit dashboard for demo/inference.

Key highlights extracted from the notebooks included in the repo:
- **Dataset size (original / used):** 4966 rows.
- **Train / Test split:** (3724, 1242) → i.e., 75% train (3724 rows) and 25% test (1242 rows).
- **Overall feature set (after preprocessing):** 119 features for the overall model.
- **Per-relay feature set:** 29 features (per relay notebook: RELAY1..RELAY4 use 29 features each).
- **Top model performances (from notebooks):**
  - **Overall model (Random Forest):** Accuracy ≈ **96.78%**
  - **Relay1 (XGBoost):** Classification report shows **macro avg ≈ 0.93** (see RELAY1 notebook for full report)
  - **Relay2 (Random Forest):** Accuracy ≈ **95.16%**
  - **Relay3 (Random Forest):** Accuracy ≈ **94.58%**
  - **Relay4 (Random Forest):** Accuracy ≈ **94.26%**

> NOTE: Exact metrics, confusion matrices and classification reports are available in the corresponding notebooks under `notebooks/` (I extracted these numbers from the notebook outputs present in the repo). If you re-run training with a different random seed, you may get slightly different metrics.

---

## Repo structure (important files)
```
Relay_signal_predictor/
├── application/
│   ├── analytics.csv              # analytics/history used by dashboard (sample)
│   └── final.py                   # Streamlit dashboard (app entrypoint)
├── models/
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   ├── random_forest_model.pkl
│   ├── random_forest_model_3.pkl
│   ├── random_forest_model_4.pkl
│   ├── xgboost.pkl
│   └── xgboost_overall_model.pkl
├── notebooks/
│   ├── overall/Model_Training_and_Testing.ipynb
│   ├── relay1/RELAY1.ipynb
│   ├── relay2/RELAY2.ipynb
│   ├── relay3/RELAY3.ipynb
│   └── relay4/RELAY4.ipynb
└── README.md (this file)
```

---

## Getting started / Reproduce locally

1. Clone the repo or download the folder and ensure this layout is preserved.

2. Create Python virtual environment and install the minimum dependencies:
```bash
python -m venv .venv
# activate .venv\Scripts\activate  (Windows)
# or source .venv/bin/activate    (Linux / macOS)
pip install -r requirements.txt   # if requirements.txt present
# Minimal requirements (if you don't have requirements.txt):
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
```

3. Place the trained model `.pkl` files (files already included under `models/`) into the working directory for the Streamlit app. The Streamlit app expects model files at the root of the project or in the same folder where `final.py` is run. You can either:
- Run from the project root (recommended) so relative paths in `application/final.py` resolve, or
- Update paths in `application/final.py` to point to `models/<modelname>.pkl`

4. Run the Streamlit dashboard (demo):
```bash
cd Relay_signal_predictor
streamlit run application/final.py
# Open the browser page it prints (usually http://localhost:8501)
```

5. If you want to re-run notebooks for training/evaluation:
```bash
jupyter notebook notebooks/overall/Model_Training_and_Testing.ipynb
# or run specific relay notebooks like:
jupyter notebook notebooks/relay1/RELAY1.ipynb
```

---

## How the dashboard (`application/final.py`) works
- The dashboard loads pre-trained models via pickle:
  - `xgboost.pkl` → Relay1
  - `random_forest_model.pkl` → Relay2
  - `random_forest_model_3.pkl` → Relay3
  - `random_forest_model_4.pkl` → Relay4
  - `xgboost_overall_model.pkl` → Overall aggregated model
- It offers separate pages for Relay1..Relay4 predictions and an Overall Prediction page that aggregates the four relays' inputs.
- It also includes analytics plots loading from `application/analytics.csv` (a simple analytics/history CSV sample shipped with the repo).

**Important:** Make sure the model files and `analytics.csv` are present in the working directory used to run Streamlit, or modify `final.py` to use absolute or `models/` paths.

---

## Dataset & Preprocessing (summary)
- The notebooks show the raw DataFrame summary: **4966 rows** and features for each relay. Preprocessing steps applied in notebooks include:
  - Dropping columns with all NaNs / zeros
  - Imputation or filtering where necessary
  - Feature extraction: time-domain features (mean, std, RMS, skew, kurtosis), frequency-domain features (FFT peaks / band power) — see the relay notebooks for exact code.
  - Scaling using `scaler.pkl` (a `StandardScaler` or similar saved artifact).
  - Label encoding using `label_encoder.pkl`

If your original raw CSV files are not included in the repo (they aren't in the zipped repo I inspected), include a small sample CSV (e.g., `data/sample_data.csv`) and a `data/README.md` instructing how to obtain the original dataset.

---

## Models (files included)
- `xgboost.pkl` — Relay1 model
- `random_forest_model.pkl` — Relay2 model
- `random_forest_model_3.pkl` — Relay3 model
- `random_forest_model_4.pkl` — Relay4 model
- `xgboost_overall_model.pkl` — Overall model that uses per-relay outputs
- `scaler.pkl`, `label_encoder.pkl` — necessary preprocessing artifacts

**Tip:** If you plan to demo live in an interview, keep a small `demo_data.csv` with 1–5 example rows and a short script `demo_predict.py` that loads a model and prints predictions. The Streamlit app already acts as an interactive demo, but a CLI script is handy for offline settings.

---

## Results (evidence from notebooks)
These numbers are taken from the notebook outputs present in the repo (see `notebooks/`):

- Dataset: **4966 rows** (checked in notebook outputs)
- Train/Test shapes: **(3724, 119)** train and **(1242, 119)** test (overall model)
- **Overall model (Random Forest)**: Accuracy ≈ **96.78%**
- **Relay1 (XGBoost)**: classification report macro avg ≈ **0.93**
- **Relay2 (Random Forest)**: Accuracy ≈ **95.16%**
- **Relay3 (Random Forest)**: Accuracy ≈ **94.58%**
- **Relay4 (Random Forest)**: Accuracy ≈ **94.26%**

For full confusion matrices and classification reports, open the corresponding notebook and find the printed `classification_report()` and `confusion_matrix()` outputs. The notebooks include plotted confusion matrices and evaluation visuals.

---

## Reproducibility checklist (make sure the repo has)
- [ ] `requirements.txt` with pinned versions (or a `environment.yml`)
- [ ] Raw datasets or a `data/README.md` describing how to obtain them
- [ ] `models/` folder with pickled models and preprocessing artifacts
- [ ] Notebooks include cells that print `df.shape` and show train/test split code
- [ ] A short `demo_data.csv` for quick manual demos

---

## What to show in an interview (60–90s demo plan)
1. Open `application/final.py` in Streamlit and run a sample input for one relay to show **real-time inference**. Explain input features and what `Attack` vs `Natural` labels mean.
2. Show the `notebooks/overall/Model_Training_and_Testing.ipynb` cell that prints **train/test shapes** and the `classification_report()` for the overall Random Forest (point to precision/recall tradeoffs).
3. Explain feature engineering choices (time vs frequency domain) and why tree models were chosen (robust to irrelevant features, fast inference).
4. If asked about deployment, explain how `final.py` loads pickled models and how you can containerize the Streamlit app with Docker for client demos.

---

## Contact / License
**Author:** Sai Kumar — https://github.com/saikumardurgavajula  

If you want the README tailored further (include exact dataset filenames, full classification reports, or a ready-made `requirements.txt` and demo script), upload the missing CSV(s) or allow me to read the notebooks directly — I can auto-fill the remaining TODOs and produce a commit-ready README.
