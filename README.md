# 🚀 DeepStack-SDP: Hybrid Software Defect Prediction Framework

DeepStack-SDP is a research-oriented hybrid Software Defect Prediction (SDP) framework designed to identify fault-prone software modules using machine learning, deep learning, ensemble learning, and explainable AI techniques.

The framework combines deep feature extraction with stacked ensemble learning to improve prediction performance while maintaining interpretability through SHAP and LIME explanations.

---

## 📌 Features

* 📂 Multi-dataset support (NASA PROMISE + defect datasets)
* 🧹 Data preprocessing and cleaning pipeline
* ⚙️ Feature engineering and selection
* ⚖️ Class imbalance handling

  * SMOTE
  * ADASYN
  * Class Weighting
* 🤖 Baseline Machine Learning Models

  * Logistic Regression
  * Random Forest
  * XGBoost
* 🧠 Deep Learning Models

  * CNN
  * LSTM
* 🔥 Proposed Hybrid Framework (**DeepStack-SDP**)

  * CNN-based feature extraction
  * Feature fusion
  * Stacking ensemble learning
  * Meta-learning with XGBoost
* 📊 Evaluation metrics

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Confusion Matrix
* 🔍 Explainable AI

  * SHAP
  * LIME
* 🌐 Streamlit interactive dashboard

---

## 🏗 Proposed DeepStack-SDP Architecture

```text
Input Dataset
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Imbalance Handling
(SMOTE / ADASYN / Class Weighting)
      ↓
CNN Feature Extraction
      ↓
Feature Fusion
(Original + Deep Features)
      ↓
Stacking Ensemble
(Random Forest + XGBoost)
      ↓
Meta Model
(XGBoost)
      ↓
Prediction
      ↓
Explainability
(SHAP + LIME)
```

---

## 📁 Project Structure

```text
software-defect-prediction-research/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│
├── notebooks/
│
├── results/
│
├── models/
│
├── reports/
│
├── streamlit_app/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── imbalance_handling/
│   ├── models/
│   │   ├── baseline_models/
│   │   ├── deep_learning_models/
│   │   └── hybrid_model/
│   │
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
│
└── README.md
```

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/your-username/DeepStack-SDP.git
cd DeepStack-SDP
```

Create environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

Train and evaluate models:

```bash
python experiments/hybrid_model_experiments.py
```

Hyperparameter tuning:

```bash
python experiments/hybrid_tuning_experiments.py
```

Generate explainability outputs:

```bash
python src/explainability/explainability_pipeline.py
```

Save trained models:

```bash
python experiments/save_models.py
```

---

## 🌐 Running Streamlit Dashboard

```bash
cd streamlit_app
streamlit run app.py
```

Dashboard Features:

* Defect prediction
* Hybrid model integration
* Model comparison leaderboard
* SHAP feature analysis
* Explainable AI visualization
* Downloadable reports

---

## 📊 Explainability Outputs

The framework generates:

* SHAP Summary Plot
* SHAP Feature Importance Plot
* SHAP Waterfall Plot
* LIME Explanation Plot
* CNN Feature Distribution Plot

Generated figures are stored in:

```text
results/
```

---

## 🧪 Technologies Used

* Python
* Scikit-learn
* TensorFlow/Keras
* XGBoost
* SHAP
* LIME
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## 📈 Research Contribution

The proposed DeepStack-SDP framework introduces:

* Deep feature extraction using CNN
* Feature fusion strategy
* Stacked ensemble learning
* Explainable AI integration
* Interactive deployment through Streamlit

This framework aims to improve software defect prediction accuracy while enhancing model interpretability.

---

## 👨‍💻 Author

Ayush Jha

Python | AI | Machine Learning Developer

---

## 📜 License

This project is intended for academic and research purposes.
