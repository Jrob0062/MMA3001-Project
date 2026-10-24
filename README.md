# MMA3001 - IoT Environmental Telemetry & Occupancy Prediction

An end-to-end data engineering and machine learning pipeline that predicts indoor room occupancy (headcount) from ambient environmental telemetry ($\text{CO}_2$, temperature, and light volatile organic compounds), benchmarked across multiple rooms and multi-year sensor deployments.

---

## 📌 Project Overview

Indoor air quality and occupancy dynamics are intrinsically linked through human metabolism: occupants continuously emit carbon dioxide ($\text{CO}_2$) and body heat. This project extracts, cleans, aligns, and models telemetry from multiple IoT sensor nodes to:
1. **Discover Physical Room Mappings**: Identify which occupancy camera shares a room with an environmental sensor using respiration physics ($\text{CO}_2$ concentration and rate-of-change correlation).
2. **Handle Asynchronous Event Alignment**: Bridge periodic environmental sensor data (recorded every 10–15 minutes) with event-driven occupancy logs (which only record when headcount changes) using chronological backward-fill (`merge_asof`).
3. **Train & Benchmark Predictive Models**: Train a **Decision Tree Regressor** on Room 869, evaluate its in-room validation performance, benchmark it against a **Linear Regression Baseline** (Topic 5.1), and test its **cross-room generalization** on unseen Room 326.
4. **Optimise & Profile Computational Complexity**: Perform a 3-fold cross-validated hyperparameter grid search and profile inference latency, memory footprint, and computational cost for edge IoT deployment.

---

## 📄 Project Documentation
The complete formal technical report (MMA3001 Data Analysis Report) can be accessed below:
* **[Download Final PDF Report](./MMA3001%20Data%20Analysis%20Report%20-%2034988041.pdf)**
* **Generated Source Documentation:** Explore the `docs/` folder for HTML-formatted code documentation.
  
---

## 🏗️ Repository Architecture

```text
MMA3001 Project/
├── 1. Given Data/                    # Raw source telemetry (gitignored due to file size)
│   ├── 5EnvSensor_MayToDec2024_180kRows.csv       # Raw nested JSON environmental logs
│   └── 5occupancySensor_MayToDec2024_9MRows.csv   # Raw 9.09M occupancy event logs
│
├── 2. Initial data analysis/         # Exploratory analysis and visual diagnostic scripts
│   ├── generate_all_plots.py
│   ├── generate_overview_plots.py
│   └── plot_sensor_temp_co2.py
│
├── 3. Initial plots/                 # Baseline IAQ and thermal comfort visualizations
│
├── 4. Reducing data Code/            # Data engineering and preprocessing pipeline
│   ├── parse_and_clean.py                         # Flattens nested JSON telemetry to tabular CSV
│   ├── Splitting data sets validation and training.py # Active window filtering & 2/3 train-val splits
│   └── Occupancy data set creation.py             # Single-pass 9M row streaming occupancy extractor
│
├── 5. Reduced Data/                  # Prepared datasets (occupancy files gitignored)
│   ├── 5EnvSensor_Cleaned_Tabular.csv             # Full cleaned environmental dataset
│   ├── 869_train.csv / 869_validation.csv         # Room 869 active period (mid-March to mid-June 2024)
│   ├── 326_train.csv / 326_validation.csv         # Room 326 active period (Oct 2025 to end of April 2026)
│   ├── 869_occupancy.csv                          # Aligned Room 869 ground-truth occupancy
│   └── 326_occupancy.csv                          # Aligned Room 326 ground-truth occupancy
│
├── 6. Decision Tree/                 # Machine learning models and evaluation
│   ├── make decision tree model.py                # Decision tree pipeline & baseline evaluation
│   ├── Decision Tree optimisation testing.py      # GridSearchCV tuning & computational profiling
│   ├── Linear Regressino Model baseline.py        # Linear regression baseline vs. Decision tree
│   └── match_rooms.py                             # Correlation analysis utility
│
├── 7. model results/                 # Exported quantitative performance metrics (CSV)
│   ├── computational_performance_and_cost.csv     # Execution times, complexity, and memory profiling
│   ├── cross_room_generalisation_results.csv      # Out-of-sample transfer performance on Room 326
│   ├── decision_tree_baseline_summary.csv         # Baseline decision tree train, val, and test metrics
│   ├── feature_importances.csv                    # Feature contribution ranking from optimal tree
│   ├── grid_search_hyperparameter_results.csv     # 3-fold CV scores for 25 hyperparameter combinations
│   ├── linear_regression_vs_decision_tree_summary.csv # Side-by-side linear regression vs tree comparison
│   └── model_performance_summary.csv              # Optimized tree metrics (MAE, RMSE, R²)
│
├── docs/                             # Interactive HTML documentation site (generated via pdoc)
│   ├── index.html                                 # Searchable API documentation portal
│   ├── Decision Tree optimisation testing.html
│   ├── Linear Regressino Model baseline.html
│   └── ... (HTML pages for all project modules)
│
└── plots/                            # High-resolution generated diagnostic figures
    ├── decision_tree_headcount_model.png          # Decision tree validation & generalization plot
    ├── linear_regression_vs_decision_tree.png     # Side-by-side baseline comparison plot
    └── decision_tree_optimisation_curve.png       # Hyperparameter tuning MAE curve
```

---

## ⚙️ Data Engineering & Active Time Windows

Telemetry devices had varying operational lifecycles, battery replacements, and sensor saturation periods. The pipeline explicitly filters for clean, non-flatlined active deployments:

| Node | Physical Active Window | Duration | Notes |
| :--- | :--- | :--- | :--- |
| **Room 869** | **15 March 2024 – 14 June 2024** | **91.2 days** | Excludes flatlined 520 ppm $\text{CO}_2$ period after mid-June |
| **Room 326** | **01 October 2025 – 22 April 2026** | **203.9 days** | Excludes inactive/flatlined 525 ppm 2024 period |

### Ground-Truth Occupancy Alignment
Because camera sensors only log events when someone enters or leaves, merging on exact timestamps drops over 95% of records. The pipeline uses **`merge_asof(direction='backward')`** to hold the last observed headcount constant until the next recorded headcount change, retaining exact integer occupant counts.

Retained occupancy fields:
* `occupancystatuschangedate` (timestamp of state change)
* `id` (unique event record ID)
* `headcount` (number of people in room)
* `occupancystatus` (`CurrentlyOccupied`, `RecentlyOccupied`, `NotOccupied`)
* `floorspaceid` (unique room identifier)

---

## 🔬 Machine Learning Performance Benchmarks

All models are trained strictly on **Room 869 Training Data** (8,107 records) and benchmarked across:
1. **Room 869 Training Set**: In-sample learning capacity.
2. **Room 869 Validation Set** (4,053 records): Unseen future time period within the same room.
3. **Room 326 Transfer Set** (26,979 records): Completely unseen room deployment across differing ventilation and physical dimensions.

### 1. Baseline Model Comparison (Linear Regression vs. Decision Tree)
*Source: [`7. model results/linear_regression_vs_decision_tree_summary.csv`](7.%20model%20results/linear_regression_vs_decision_tree_summary.csv)*

| Model | Evaluation Dataset | Sample Count | MAE [People] | RMSE [People] | $R^2$ Score | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression Baseline** | Room 869 Train | 8,107 | 0.60 | 0.80 | +0.343 | Continuous plane cannot capture zero-inflation |
| **Decision Tree Regressor** | Room 869 Train | 8,107 | **0.51** | **0.75** | **+0.429** | **15.0% lower MAE**; captures non-linear step shifts |
| **Linear Regression Baseline** | Room 869 Validation | 4,053 | 0.57 | 0.75 | +0.176 | Frequently predicts negative headcounts ($<0$) |
| **Decision Tree Regressor** | Room 869 Validation | 4,053 | **0.49** | **0.71** | **+0.256** | **14.0% lower MAE**; naturally bounds predictions $\ge 0$ |
| **Linear Regression Baseline** | Room 326 Generalisation | 26,979 | 1.07 | 1.92 | -0.283 | Degrades significantly under baseline drift |
| **Decision Tree Regressor** | Room 326 Generalisation | 26,979 | **0.65** | **1.72** | **-0.032** | **39.2% lower MAE**; robust cross-room transfer |

---

## 📈 Model Optimisation & Grid Search Results

To eliminate overfitting and optimize the bias-variance trade-off, a 3-fold cross-validated grid search was executed across 25 parameter configurations:
* `max_depth` $\in \{3, 5, 8, 12, \text{None}\}$
* `min_samples_leaf` $\in \{5, 10, 15, 25, 50\}$

*Source: [`7. model results/grid_search_hyperparameter_results.csv`](7.%20model%20results/grid_search_hyperparameter_results.csv)*

### Top Hyperparameter Configurations (3-Fold CV)

| Rank | `max_depth` | `min_samples_leaf` | Mean CV MAE [People] | Std CV MAE | Mean Fit Time [s] | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **5** | **10** | **0.5665** | **$\pm 0.0577$** | **0.020s** | **Optimal balance of depth and leaf regularization** |
| 2 | 3 | 10 | 0.5678 | $\pm 0.0643$ | 0.012s | Shallow tree; low variance but slight underfit |
| 7 | 12 | 50 | 0.5682 | $\pm 0.0642$ | 0.022s | High depth tempered by strong leaf regularization |
| 14 | 8 | 25 | 0.5796 | $\pm 0.0657$ | 0.022s | Intermediate performance |
| 24 | 12 | 5 | 0.6303 | $\pm 0.0446$ | 0.032s | Overfitting to local sensor fluctuations |
| 25 | None | 5 | 0.6504 | $\pm 0.0400$ | 0.033s | Worst CV score; unconstrained depth memorizes training noise |

> **Key Insight:** Unconstrained trees (`max_depth=None`, `min_samples_leaf=5`) suffer an error increase of nearly **15%** on unseen cross-validation splits compared to the regularized model (`max_depth=5`, `min_samples_leaf=10`), proving that depth pruning and leaf-size constraints are critical for indoor environmental telemetry.

---

## 🔍 Feature Importance Breakdown

*Source: [`7. model results/feature_importances.csv`](7.%20model%20results/feature_importances.csv)*

The optimal tree's splits are overwhelmingly dominated by metabolic emission dynamics and daily circadian schedules:

| Feature Name | Relative Importance | Percentage | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| **`co2_ppm`** | **0.5358** | **53.58%** | Primary metabolic indicator; direct accumulation of exhaled human breath |
| **`hour`** | **0.3454** | **34.54%** | Circadian schedule indicator; differentiates work hours from nighttime closures |
| **`co2_ppm_roll3`** | **0.0576** | **5.76%** | Rolling 30–45 min trend; smooths transient sensor fluctuations |
| **`day_of_week`** | **0.0417** | **4.17%** | Separates high-occupancy mid-week days from low-occupancy Fridays |
| **`temperature_c`** | **0.0052** | **0.52%** | Ambient thermal rise from sensible body heat emission |
| **`temperature_c_lag1`** | **0.0049** | **0.49%** | Short-term thermal inertia |
| **`lvoc_ppb`** | **0.0036** | **0.36%** | Volatile organic compounds from occupants and materials |
| **`lvoc_ppb_roll3`** | **0.0036** | **0.36%** | Rolling VOC concentration trend |
| *Others (lags/diffs)* | <0.0020 | <0.22% | Marginal contribution to tree splits |

---

## ⚡ Computational Cost & Edge Deployment Profiling

*Source: [`7. model results/computational_performance_and_cost.csv`](7.%20model%20results/computational_performance_and_cost.csv)*

The Decision Tree was profiled to verify feasibility for real-time inference on resource-constrained embedded microcontrollers (e.g., ESP32, Raspberry Pi Zero) at the building edge:

| Profiling Metric | Measured Value | Practical Edge Feasibility |
| :--- | :--- | :--- |
| **Total Pipeline Runtime** | **10.05 seconds** | Includes CSV parsing, feature engineering, 75-fit grid search, and exports |
| **Feature Engineering & Alignment** | **0.37 seconds** | Vectorized NumPy/Pandas operations |
| **Optimal Model Training Time** | **0.0218 seconds** | Fast retraining cycle allows daily on-device recalibration |
| **Inference Time (39,139 rows)** | **0.0094 seconds** | **~0.24 microseconds per prediction** |
| **Inference Throughput** | **4,176,590 rows/second** | Capable of real-time multi-room building management |
| **Memory Footprint** | **~18.5 MB** | Readily fits within small edge gateway RAM budgets |
| **Training Complexity** | **$O(N \log N \cdot M)$** | Scalable with training size $N=8,107$, features $M=15$ |
| **Inference Complexity** | **$O(\text{max\_depth}) = O(5)$** | Fixed 5 binary if-else comparisons per reading |

---

## 📊 Visualizations

All diagnostic plots are automatically generated and saved in the [`plots/`](plots/) folder:

1. **Hyperparameter Tuning Curve**: `plots/decision_tree_optimisation_curve.png`
   * Illustrates 3-fold cross-validation MAE across varying tree depths and leaf sizes, highlighting the optimal `depth=5` minimum.
2. **Model Evaluation & Generalisation**: `plots/decision_tree_headcount_model.png`
   * Displays feature importance rankings and 2-week actual vs. predicted headcount time series for both Room 869 validation and Room 326 out-of-sample transfer.
3. **Linear Regression Baseline vs. Decision Tree**: `plots/linear_regression_vs_decision_tree.png`
   * Side-by-side MAE bar chart and zoomed time series highlighting the Decision Tree's reduction of false negative predictions.

---

## 🚀 Running the Pipeline Locally

### 1. Requirements
Ensure Python 3.9+ is installed, then install project dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib pdoc
```

### 2. Preprocessing & Data Splits
To reproduce the clean train/validation and occupancy datasets:
```bash
# 1. Split active environmental data for Room 869 and Room 326
python "4. Reducing data Code/Splitting data sets validation and training.py"

# 2. Extract matching occupancy datasets from raw 9M-row log
python "4. Reducing data Code/Occupancy data set creation.py"
```

### 3. Model Training, Optimisation & Results Generation
```bash
# 1. Run baseline Decision Tree Regressor
python "6. Decision Tree/make decision tree model.py"

# 2. Run hyperparameter grid search, computational profiling & export all CSV tables
python "6. Decision Tree/Decision Tree optimisation testing.py"

# 3. Run comparative benchmark against Linear Regression baseline
python "6. Decision Tree/Linear Regressino Model baseline.py"
```

All quantitative outputs are exported to CSV files in `7. model results/`, and figures are saved to `plots/`.

---

## 📖 Interactive HTML Documentation (`pdoc`)

The codebase adheres strictly to professional Python software engineering and documentation standards:
* **Google-Style Docstrings**: Every module, class, and function provides a structured narrative, typed argument specifications (`Args:`), return value descriptions (`Returns:`), anticipated exceptions (`Raises:`), and executable usage examples (`Examples:`).
* **Intent-Driven Inline Comments**: Comments clarify physical domain intent (e.g., metabolic exhalation rates, sensor drift, Nyquist intervals) rather than repeating Python syntax.
* **No Side-Effects on Import**: All scripts encapsulate executable logic inside functions guarded by `if __name__ == '__main__':`, enabling clean import and automated introspection.

### 🌐 Viewing or Regenerating HTML Documentation
To browse the documentation site locally:
* Open [`docs/index.html`](docs/index.html) directly in any standard web browser.
* Or launch `pdoc`'s live local server:
  ```bash
  python -m pdoc "6. Decision Tree/Decision Tree optimisation testing.py" "6. Decision Tree/Linear Regressino Model baseline.py" "6. Decision Tree/make decision tree model.py"
  ```
To regenerate the standalone HTML distribution:
```bash
python -m pdoc -o docs -d google --math "6. Decision Tree/Decision Tree optimisation testing.py" "6. Decision Tree/Linear Regressino Model baseline.py" "6. Decision Tree/make decision tree model.py" "6. Decision Tree/match_rooms.py" "4. Reducing data Code/Splitting data sets validation and training.py" "4. Reducing data Code/Occupancy data set creation.py" "4. Reducing data Code/parse_and_clean.py" "4. Reducing data Code/split 869 data.py" "4. Reducing data Code/master data set.py" "4. Reducing data Code/compare_869_march_to_may.py" "2. Initial data analysis/generate_all_plots.py" "2. Initial data analysis/generate_overview_plots.py" "2. Initial data analysis/plot_sensor_temp_co2.py"
```
