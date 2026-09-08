# 🏦 Bank Customer Churn AI & Retention Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB.svg?logo=react&logoColor=black)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB5424.svg)](https://xgboost.readthedocs.io/)

A full-stack, production-grade Machine Learning SaaS application designed for retail banks and financial institutions to predict customer churn, explain risk factors with Explainable AI (XAI), estimate financial exposure with Customer Lifetime Value (CLV) modeling, and simulate retention actions in real time.

---

## 🌟 Key Highlights & Design System

- **🎨 Modern Cotton Candy Pastel SaaS UI**: Clean, accessible pastel gradient aesthetic (`#FFDEE9` Soft Pink to `#B5FFFC` Mint Blue) paired with `#2C3E50` Dark Slate Blue typography for maximum legibility.
- **🪄 3D Micro-Interactions & Animations**: Interactive 3D card tilt on hover (`perspective(1000px)`), 3D floating header levitation (`@keyframes float3D`), and celebratory confetti for major risk-reduction milestones.
- **⚡ Decoupled Full-Stack Architecture**: Modern **React.js (Vite)** frontend powered by a high-throughput **FastAPI (Python)** REST backend, alongside an optional **Streamlit** dashboard.
- **🌍 Global Demographics & Geography**: Built-in support for **India 🇮🇳, United States 🇺🇸, United Kingdom 🇬🇧, Canada 🇨🇦, Australia 🇦🇺, France 🇫🇷, Germany 🇩🇪, and Spain 🇪🇸**.

---

## 📁 Project Architecture & Directory Structure

```text
bank-churn-prediction/
│
├── backend/
│   └── server.py                     # FastAPI REST API serving inference & analytics
│
├── frontend/                         # React.js (Vite) SaaS Web Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx           # Navigation, model switcher & threshold slider
│   │   │   ├── GaugeChart.jsx        # Animated 3D SVG Semicircle Radial Gauge
│   │   │   ├── SinglePredictor.jsx   # Single customer scoring, XAI & What-If simulator
│   │   │   ├── BatchPredictor.jsx    # CSV batch evaluation & portfolio valuation
│   │   │   └── ModelPerformance.jsx  # Multi-model benchmarks & feature importances
│   │   ├── api.js                    # Centralized API service layer
│   │   ├── App.jsx                   # Main React component
│   │   └── index.css                 # Cotton Candy theme & 3D animation stylesheet
│   ├── package.json
│   └── vite.config.js
│
├── src/                              # Core Machine Learning & Analytics Pipeline
│   ├── data_download.py              # Automated raw dataset retrieval
│   ├── data_preprocessing.py         # One-Hot Encoding & StandardScaler pipelines
│   ├── train_model.py                # Model training (ANN, Random Forest, XGBoost, LR)
│   ├── evaluate_model.py             # Metrics calculation (Acc, Prec, Rec, F1, AUC)
│   └── predict.py                    # Inference engine, CLV, XAI, What-If simulator
│
├── models/                           # Serialized Trained Model Artifacts
│   ├── preprocessor.pkl              # Scikit-learn ColumnTransformer pipeline
│   ├── churn_model.keras             # Trained Deep Learning Artificial Neural Network
│   ├── random_forest.pkl             # Trained Random Forest Classifier
│   ├── xgboost.pkl                   # Trained XGBoost Classifier
│   ├── logistic_regression.pkl       # Trained Logistic Regression Classifier
│   └── model_comparison.json         # Compiled evaluation metrics across all models
│
├── data/
│   └── Churn_Modelling.csv           # 10,000 customer banking records
│
├── app/
│   └── app.py                        # Alternative Streamlit interactive web dashboard
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Comprehensive documentation
```

---

## 🔬 In-Depth Feature Breakdown & Technical Mechanics

### 1. 🔮 Single Customer Assessment & 3D Radial Gauge Meter
* **How It Works**: 
  - Collects customer attributes: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
  - Passes features through `preprocessor.pkl` to scale numerical attributes via `StandardScaler` and encode categorical values via `OneHotEncoder(handle_unknown='ignore')`.
  - Generates the continuous churn probability $P(\text{Churn}) \in [0, 1]$ from the selected classifier (`ANN`, `Random Forest`, `XGBoost`, or `Logistic Regression`).
  - Dynamically renders an animated 3D SVG Semicircle Radial Gauge with colored risk bands reflecting the active decision threshold.

---

### 2. 🔍 Explainable AI (XAI) & Factor Attribution Engine
* **How It Works**:
  - Compares the customer's specific attributes against cohort statistical benchmarks.
  - Highlights top **Risk Drivers** (factors pushing churn probability up) and **Protective Factors** (factors anchoring customer retention).
  - **Sample Risk Factor Logic**:
    - **Age $\ge 50$**: Identifies higher historical churn propensity in older demographics ($+28\%$ risk weighting).
    - **Inactive Status**: Inactive accounts lack engagement and represent primary switching risks ($+24\%$ risk weighting).
    - **Single Product ($1$)**: Customers with only one product have low switching barriers ($+18\%$ risk weighting).
    - **Multi-Product Loyalty ($2$)**: Two products represent the empirical retention sweet spot ($\Delta -25\%$ risk reduction).
    - **Geography (India 🇮🇳)**: Recognizes high digital banking stickiness and long-term brand loyalty ($\Delta -12\%$ risk reduction).

---

### 3. 💡 Interactive "What-If" Retention Simulator
* **How It Works**:
  - Relationship managers can test proposed retention interventions (e.g. converting an Inactive Member $\rightarrow$ Active, issuing a 2nd product, or providing a Credit Card).
  - The engine clones the customer state, applies proposed adjustments, re-evaluates the ML model, and computes:
    $$\Delta \text{ Risk Reduction (\%)} = (P_{\text{simulated}} - P_{\text{original}}) \times 100$$
  - Displays the transition across risk tiers (e.g., **High Risk $\rightarrow$ Low Risk**) and fires interactive celebratory confetti whenever retention actions yield $\ge 20\%$ risk reduction.

---

### 4. 💰 Customer Lifetime Value (CLV) & Financial Risk Engine
* **How It Works**:
  - Models annual revenue generated by a retail banking customer using standard financial formulas:
    $$\text{Annual Revenue} = \left( \text{Balance} \times 2.5\% + \text{Products} \times \$150 + \text{Card Fee} \right) \times \text{Active Multiplier}$$
  - Computes remaining customer lifetime horizon $T_{\text{future}} = \max(2.0, \min(10.0, 7.5 - \text{Tenure} \times 0.2))$.
  - Derives total **Customer Lifetime Value (CLV)**:
    $$\text{CLV} = \text{Annual Revenue} \times T_{\text{future}}$$
  - Identifies **Portfolio Balance at Risk** and **Annual Revenue at Risk** for at-risk accounts.

---

### 5. 📂 Batch CSV Predictions & Portfolio Risk Valuation
* **How It Works**:
  - Allows bulk evaluation of thousands of customer records at once via multipart CSV file upload or a 1-click **"Load Sample Test Data"** button.
  - Returns portfolio-wide macro KPIs:
    - **Total Customers Evaluated**
    - **High Churn Risk Customers Count & Churn Rate %**
    - **Total Balance at Risk ($)**
    - **Total Annual Revenue at Risk ($)**
    - **Projected Campaign Savings ($)** (assuming standard $20\%$ retention intervention rescue rate).
  - Provides a filterable, styled predictions table with one-click full CSV export.

---

### 6. 🎯 Dynamic Decision Threshold ($\Theta$) Slider
* **How It Works**:
  - In standard ML models, classification defaults to $\Theta = 0.50$.
  - In retail banking, **Recall** is more critical than Precision: failing to catch a churning customer permanently loses their balance and CLV, whereas false alarms only incur the minor cost of a retention email or discount offer.
  - The sidebar slider ($\Theta \in [0.10, 0.90]$) enables relationship managers to adjust sensitivity:
    - **$\Theta < 0.50$ (High-Recall Mode)**: Proactively catches potential churners earlier.
    - **$\Theta > 0.50$ (High-Precision Mode)**: Focuses exclusively on high-certainty churners.

---

### 7. 📄 Executive Customer Churn Report Generator
* **How It Works**:
  - Formats a confidential executive memorandum (`.txt`) for relationship managers containing customer demographic summaries, AI churn assessments, CLV metrics, primary XAI drivers, and prioritized retention recommendations.

---

### 8. 📊 Multi-Model Performance Benchmarks & Tree Feature Importance
* **How It Works**:
  - Evaluates and compares all 4 trained models on identical stratified test splits across 5 standard metrics:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-Score**
    - **ROC-AUC**
  - Extracts and charts global **Mean Decrease in Impurity (MDI)** from Random Forest and **Feature Gain** from XGBoost.

---

## 🤖 Deep Learning ANN Architecture

The deep learning model is built using **TensorFlow / Keras**:
- **Input Layer**: 12 preprocessed features (including one-hot encoded country and gender columns).
- **Dense Layer 1**: 64 neurons, ReLU activation function.
- **Dropout Layer**: $20\%$ dropout rate to mitigate overfitting.
- **Dense Layer 2**: 32 neurons, ReLU activation function.
- **Dropout Layer**: $20\%$ dropout rate.
- **Dense Layer 3**: 16 neurons, ReLU activation function.
- **Output Layer**: 1 neuron, Sigmoid activation (outputs churn probability $P \in [0, 1]$).
- **Loss Function**: Binary Cross-Entropy with balanced class weighting.
- **Optimizer**: Adam with Early Stopping and model checkpointing.

---

## 🚀 Quickstart & Installation Guide

### Prerequisites
- **Python 3.9+**
- **Node.js 18+** and **npm**

---

### 1. Clone & Set Up the Python Environment

```bash
# Clone repository
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

---

### 2. Run the Machine Learning Pipeline (Optional if models already saved)

```bash
# 1. Download dataset
python src/data_download.py

# 2. Train all 4 ML models and save preprocessors
python src/train_model.py

# 3. Evaluate models and output classification reports
python src/evaluate_model.py
```

---

### 3. Start the FastAPI Backend Server

```bash
python -m uvicorn backend.server:app --port 8000 --host 0.0.0.0 --reload
```
* Backend will be live at: **`http://localhost:8000`**
* Interactive Swagger API Docs: **`http://localhost:8000/docs`**

---

### 4. Start the React.js Frontend Application

Open a second terminal window:

```bash
cd frontend
npm install
npm run dev
```
* React Web App will be live at: **`http://localhost:5173`**

---

### 5. Alternative: Run the Streamlit Dashboard

If you prefer running the Streamlit dashboard:

```bash
streamlit run app/app.py
```
* Streamlit App will be live at: **`http://localhost:8501`**

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health check |
| `POST` | `/api/predict/single` | Computes churn probability, CLV, and XAI factor attribution |
| `POST` | `/api/predict/what-if` | Simulates customer retention adjustments and calculates $\Delta\%$ |
| `POST` | `/api/predict/batch` | Uploads CSV and computes batch predictions & portfolio risk valuation |
| `GET` | `/api/data/sample-csv` | Generates 50 synthetic test customer records as downloadable CSV |
| `GET` | `/api/models/comparison` | Returns benchmark performance metrics for all 4 trained models |
| `GET` | `/api/models/feature-importance` | Returns global tree feature importance rankings (RF & XGBoost) |
| `GET` | `/api/eda/demographics` | Returns summarized demographic statistics |

---

## 🛡️ License
This project is open-source under the MIT License.
