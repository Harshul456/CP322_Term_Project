# Complete Pipeline Explanation - CP322 Regression Project

## 📋 Table of Contents
1. [Overall Architecture](#overall-architecture)
2. [Entry Point: main.py](#entry-point-mainpy)
3. [Module-by-Module Breakdown](#module-by-module-breakdown)
4. [Data Flow Through Pipeline](#data-flow-through-pipeline)
5. [Model Training Process](#model-training-process)
6. [Results Organization](#results-organization)

---

## 🏗️ Overall Architecture

The project follows a **modular pipeline architecture**:

```
main.py (Orchestrator)
    ↓
    ├── Data Loading (data_loader.py)
    ├── Preprocessing (preprocessing.py)
    ├── Model Training (baselines.py, xgboost_model.py, tabnet_model.py)
    ├── Hyperparameter Tuning (tuning.py)
    ├── SHAP Analysis (shap_analysis.py)
    └── Results Saving (utils.py)
```

**Key Principle**: Each module has a single responsibility and can be used independently.

---

## 🚀 Entry Point: main.py

### Purpose
The main orchestrator that runs the complete pipeline for both datasets.

### How It Works

#### Step 1: Initialization
```python
# Sets random seed for reproducibility
set_seed(42)

# Defines both datasets
datasets = [
    {"name": "EnergyEfficiency", "path": "...", "target": "Heating_Load"},
    {"name": "EnergyAppliances", "path": "...", "target": "Appliances"}
]
```

#### Step 2: Pipeline Execution
For each dataset, calls `run_pipeline()` which:

1. **Loads Data** → `load_dataset()`
2. **Preprocesses** → `preprocess_data()`
3. **Trains Models** → Multiple training functions
4. **Tunes Hyperparameters** → `tune_xgboost()`
5. **Generates SHAP Plots** → `create_shap_plots()`
6. **Saves Results** → `save_json()`

#### Step 3: Summary
After both datasets complete, saves a summary JSON with all metrics.

---

## 📦 Module-by-Module Breakdown

### 1. **src/utils.py** - Utility Functions

**Purpose**: Helper functions used throughout the project.

#### Functions:

**`set_seed(seed=42)`**
- Sets random seeds for Python, NumPy, PyTorch, and CUDA
- Ensures reproducible results across runs
- Called once at the start of `main()`

**`save_json(data, filepath)`**
- Saves dictionaries to JSON files
- Creates directories if they don't exist
- Used to save metrics and tuning results

**`load_json(filepath)`**
- Loads JSON files back into Python dictionaries
- Used for reading saved results

---

### 2. **src/data_loader.py** - Data Loading

**Purpose**: Simple CSV file loader.

#### Function: `load_dataset(filepath)`

**What it does:**
1. Checks if file exists
2. Uses pandas to read CSV
3. Prints dataset info (shape, columns)
4. Returns pandas DataFrame

**Example:**
```python
df = load_dataset("data/energy_efficiency_data.csv")
# Returns: DataFrame with all columns
```

**Output:**
- Shape: (768, 10) for Energy Efficiency
- Shape: (19735, 29) for Energy Appliances

---

### 3. **src/preprocessing.py** - Data Preprocessing

**Purpose**: Transforms raw data into model-ready format.

#### Function: `identify_column_types(df, target_col)`

**What it does:**
1. Separates target column from features
2. Categorizes each column:
   - **Date columns**: Detects datetime strings (tries parsing)
   - **Categorical columns**: Object types OR numeric with <20 unique values
   - **Numeric columns**: Everything else
3. **Special handling**: Drops high-cardinality columns (>100 unique values) to prevent feature explosion

**Returns:** `(categorical_cols, numeric_cols, date_cols)`

#### Function: `preprocess_data(df, target_col, test_size=0.2, random_state=42)`

**Step-by-step process:**

1. **Separate Features and Target**
   ```python
   X = df.drop(columns=[target_col])  # Features
   y = df[target_col]                  # Target variable
   ```

2. **Handle Date Columns**
   - Converts date strings to datetime objects
   - Extracts: year, month, day, dayofweek, hour
   - Creates new numeric features (e.g., `date_year`, `date_month`)
   - Drops original date column

3. **Handle Missing Values**
   - Numeric columns: Fill with median
   - Categorical columns: Fill with mode

4. **Create Preprocessing Pipeline**
   ```python
   ColumnTransformer([
       ('num', StandardScaler(), numeric_cols),      # Scale numeric features
       ('cat', OneHotEncoder(...), categorical_cols)  # One-hot encode categorical
   ])
   ```

5. **Split Data**
   - 80% training, 20% test
   - Uses `random_state=42` for reproducibility

6. **Transform Data**
   - Fit preprocessor on training data
   - Transform both train and test sets
   - Converts back to DataFrames with feature names

**Returns:** `(X_train, X_test, y_train, y_test, preprocessor)`

**Example Transformation:**
- Input: 8 categorical columns
- Output: 44 features (after one-hot encoding)

---

### 4. **src/baselines.py** - Baseline Models

**Purpose**: Simple baseline models for comparison.

#### Function: `evaluate_model(y_true, y_pred, model_name)`

**What it does:**
- Calculates three metrics:
  - **RMSE** (Root Mean Squared Error): Lower is better
  - **MAE** (Mean Absolute Error): Average prediction error
  - **R²** (Coefficient of Determination): 1.0 = perfect, 0.0 = baseline
- Returns dictionary of metrics

#### Function: `train_linear_regression(...)`

**Process:**
1. **Checks for Issues:**
   - If features > samples → Uses Ridge Regression (regularization)
   - If features > 10,000 → Uses Ridge Regression
   - Otherwise → Uses standard Linear Regression

2. **Data Validation:**
   - Checks for NaN values → Fills with 0
   - Checks for infinite values → Replaces with 0

3. **Training:**
   ```python
   model.fit(X_train, y_train)  # Learns coefficients
   ```

4. **Prediction & Evaluation:**
   ```python
   y_pred = model.predict(X_test)
   metrics = evaluate_model(y_test, y_pred, "Linear Regression")
   ```

5. **Saving:**
   - Saves model as `.joblib` file
   - Returns metrics dictionary

**How Linear Regression Works:**
- Finds best line: `y = w₁x₁ + w₂x₂ + ... + b`
- Minimizes sum of squared errors
- Very fast (< 1 second)

#### Function: `train_random_forest(...)`

**Process:**
1. **Creates Model:**
   ```python
   RandomForestRegressor(n_estimators=100, random_state=42)
   ```
   - Builds 100 decision trees
   - Each tree votes on prediction
   - Final prediction = average of all trees

2. **Training:**
   - Each tree trained on random subset of data (bootstrap)
   - Each split uses random subset of features
   - Prevents overfitting

3. **Prediction & Evaluation:**
   - Same as Linear Regression

**How Random Forest Works:**
- Ensemble of decision trees
- More robust than single tree
- Handles non-linear relationships

---

### 5. **src/xgboost_model.py** - XGBoost Model

**Purpose**: Advanced gradient boosting model.

#### Function: `train_xgboost(...)`

**Process:**
1. **Creates XGBoost Model:**
   ```python
   XGBRegressor(
       n_estimators=100,      # Number of trees
       max_depth=6,            # Tree depth
       learning_rate=0.1,      # Step size
       subsample=0.8,          # Row sampling
       colsample_bytree=0.8    # Column sampling
   )
   ```

2. **Training:**
   - Gradient boosting: Sequentially adds trees
   - Each new tree corrects errors of previous trees
   - Uses gradient descent to minimize loss

3. **How XGBoost Works:**
   - Starts with initial prediction (mean)
   - Adds trees one by one
   - Each tree predicts the "residual" (error)
   - Final prediction = sum of all tree predictions
   - Example: `prediction = 50 + tree1(-5) + tree2(+2) + ... = 47`

4. **Returns:** Both metrics and model (needed for SHAP)

---

### 6. **src/tabnet_model.py** - TabNet Neural Network

**Purpose**: Deep learning model specifically designed for tabular data.

#### Function: `train_tabnet(...)`

**Process:**
1. **Data Conversion:**
   - Converts to NumPy arrays
   - Converts to float32 (faster training)
   - Reshapes target to 2D: `(n_samples, 1)`

2. **Creates TabNet Model:**
   ```python
   TabNetRegressor(
       n_d=64, n_a=64,        # Hidden dimensions
       n_steps=5,             # Decision steps
       gamma=1.5,             # Relaxation parameter
       lambda_sparse=1e-3,    # Sparsity regularization
       optimizer_fn=Adam,     # Optimizer
       learning_rate=2e-2     # Learning rate
   )
   ```

3. **Training Process:**
   - **Epochs**: One complete pass through training data
   - **Batch Size**: Processes 1024 samples at a time
   - **Early Stopping**: Stops if validation RMSE doesn't improve for 5 epochs
   - **Validation**: Monitors test set performance during training

4. **How TabNet Works:**
   - Neural network with attention mechanism
   - Learns which features to focus on (feature selection)
   - Processes data in "steps" (like decision trees)
   - Uses sparsity to select important features

5. **Training Output:**
   ```
   epoch 0  | loss: 16336.59 | val_0_rmse: 107.85
   epoch 1  | loss: 10577.99 | val_0_rmse: 99.17
   ...
   Early stopping at epoch 15
   ```

---

### 7. **src/tuning.py** - Hyperparameter Tuning

**Purpose**: Finds best hyperparameters using Optuna (Bayesian optimization).

#### Function: `objective(trial, ...)`

**What it does:**
- **Trial**: Optuna suggests hyperparameter values
- **Training**: Trains XGBoost with suggested parameters
- **Evaluation**: Calculates RMSE on validation set
- **Returns**: RMSE (Optuna minimizes this)

**Hyperparameters Tuned:**
- `n_estimators`: 50-500 (number of trees)
- `max_depth`: 3-10 (tree depth)
- `learning_rate`: 0.01-0.3 (step size)
- `subsample`: 0.6-1.0 (row sampling)
- `colsample_bytree`: 0.6-1.0 (column sampling)

#### Function: `tune_xgboost(...)`

**Process:**
1. **Split Training Data:**
   - 80% for training, 20% for validation
   - Test set kept separate (final evaluation only)

2. **Create Optuna Study:**
   ```python
   study = optuna.create_study(
       direction='minimize',  # Minimize RMSE
       sampler=TPESampler()   # Tree-structured Parzen Estimator
   )
   ```

3. **Optimization Loop:**
   - Runs 50 trials
   - Each trial: Suggests parameters → Trains → Evaluates
   - Tracks best parameters found so far

4. **How Optuna Works:**
   - **Bayesian Optimization**: Uses past results to suggest better parameters
   - **TPE Sampler**: Models probability distribution of good parameters
   - **Smart Search**: Focuses on promising regions of parameter space

5. **Final Model:**
   - Trains on **full training set** with best parameters
   - Evaluates on **test set** (unseen data)
   - Saves best parameters to JSON

**Example Output:**
```
Best parameters: {
    'n_estimators': 441,
    'max_depth': 8,
    'learning_rate': 0.0176,
    'subsample': 0.97,
    'colsample_bytree': 0.65
}
Best RMSE: 0.7188
Test RMSE: 0.7095
```

---

### 8. **src/shap_analysis.py** - SHAP Feature Importance

**Purpose**: Explains model predictions using SHAP (SHapley Additive exPlanations).

#### Function: `create_shap_plots(...)`

**Process:**
1. **Prepare Data:**
   - Converts to NumPy arrays
   - Uses sample of 100 test instances (for speed)

2. **Create Explainer:**
   - **First tries**: `shap.Explainer()` (auto-selects best method)
   - **Falls back to**: `shap.PermutationExplainer()` (works with any model)
   - Handles XGBoost version compatibility issues

3. **Calculate SHAP Values:**
   - For each sample, calculates contribution of each feature
   - SHAP values show: "How much did this feature change the prediction?"

4. **Generate Plots:**

   **Summary Plot:**
   - Shows feature importance
   - Color = feature value (red = high, blue = low)
   - Position = SHAP value (right = increases prediction)
   - Example: High temperature → increases energy prediction

   **Bar Plot:**
   - Mean absolute SHAP values
   - Features sorted by importance
   - Shows which features matter most overall

5. **Save Plots:**
   - Saves as PNG files (300 DPI, high quality)
   - One summary plot + one bar plot per model

**How SHAP Works:**
- Based on game theory (Shapley values)
- For each prediction, calculates: "What if this feature wasn't present?"
- SHAP value = average contribution across all feature combinations
- Positive SHAP = feature increases prediction
- Negative SHAP = feature decreases prediction

---

## 🔄 Data Flow Through Pipeline

### Complete Flow Diagram:

```
1. main.py starts
   ↓
2. set_seed(42) - Ensures reproducibility
   ↓
3. For each dataset:
   ↓
   4. load_dataset() 
      Input: "data/energy_efficiency_data.csv"
      Output: DataFrame (768 rows × 10 columns)
      ↓
   5. preprocess_data()
      Input: Raw DataFrame
      Steps:
        - Separate X (features) and y (target)
        - Convert dates → numeric features
        - Handle missing values
        - Identify column types
        - Create preprocessing pipeline
        - Split train/test (80/20)
        - Fit & transform
      Output: 
        - X_train (614 rows × 44 features)
        - X_test (154 rows × 44 features)
        - y_train, y_test
        - preprocessor object
      ↓
   6. train_linear_regression()
      Input: X_train, y_train, X_test, y_test
      Process:
        - Create LinearRegression model
        - model.fit(X_train, y_train)
        - y_pred = model.predict(X_test)
        - Calculate RMSE, MAE, R²
        - Save model to .joblib
      Output: metrics dict
      ↓
   7. train_random_forest()
      Input: X_train, y_train, X_test, y_test
      Process: Similar to Linear Regression
      Output: metrics dict
      ↓
   8. train_xgboost()
      Input: X_train, y_train, X_test, y_test
      Process: Gradient boosting training
      Output: metrics dict + model object
      ↓
   9. train_tabnet()
      Input: X_train, y_train, X_test, y_test
      Process: Neural network training (epochs)
      Output: metrics dict + model object
      ↓
   10. tune_xgboost()
       Input: X_train, y_train, X_test, y_test
       Process:
         - Split train into train/val (80/20)
         - Run 50 Optuna trials
         - Each trial: Train → Evaluate → Suggest better params
         - Train final model with best params
         - Evaluate on test set
       Output: best_params dict + tuned model
       ↓
   11. create_shap_plots()
       Input: model, X_test, feature_names
       Process:
         - Sample 100 test instances
         - Calculate SHAP values
         - Generate summary plot
         - Generate bar plot
         - Save PNG files
       Output: SHAP plots saved to disk
       ↓
   12. save_json()
       Input: all_metrics dictionary
       Output: results/{dataset}/metrics/all_metrics.json
```

---

## 🎯 Model Training Process (Detailed)

### Linear Regression Training:

```
1. Check data shape
   - If features > samples → Use Ridge (regularization)
   
2. Validate data
   - Remove NaN values
   - Remove infinite values
   
3. Fit model
   - Solves: min ||y - Xw||²
   - Finds optimal weights w
   - Time: < 1 second
   
4. Predict
   - y_pred = X_test @ w + bias
   
5. Evaluate
   - Compare y_pred vs y_test
   - Calculate RMSE, MAE, R²
```

### Random Forest Training:

```
1. Create 100 decision trees
   
2. For each tree:
   - Sample random subset of data (bootstrap)
   - Build tree with random feature subsets
   - Make predictions
   
3. Aggregate predictions
   - Average all tree predictions
   - Final prediction = mean(tree1, tree2, ..., tree100)
   
4. Evaluate
   - Same metrics as Linear Regression
```

### XGBoost Training:

```
1. Initialize
   - Start with mean(y_train) as initial prediction
   
2. For each tree (1 to 100):
   a. Calculate residuals (errors)
      residual = y_train - current_prediction
   
   b. Build tree to predict residuals
      - Uses gradient descent
      - Finds splits that minimize loss
   
   c. Update prediction
      prediction = prediction + learning_rate * tree_prediction
   
3. Final prediction
   - Sum of all tree contributions
   - prediction = initial + tree1 + tree2 + ... + tree100
```

### TabNet Training:

```
1. Initialize neural network
   - Random weights
   
2. For each epoch (1 to 50, or until early stopping):
   a. Forward pass
      - Input: X_train batch
      - Through network layers
      - Output: predictions
   
   b. Calculate loss
      - MSE between predictions and y_train
   
   c. Backward pass (backpropagation)
      - Calculate gradients
      - Update weights using Adam optimizer
   
   d. Validate
      - Check performance on X_test
      - If no improvement for 5 epochs → stop early
   
3. Use best weights (from best epoch)
```

### Optuna Tuning:

```
1. Split training data
   - 80% train, 20% validation
   
2. Initialize study
   - TPE Sampler (Bayesian optimization)
   
3. For each trial (1 to 50):
   a. Suggest hyperparameters
      - Based on previous trial results
      - Focuses on promising regions
   
   b. Train XGBoost with suggested params
      - Train on 80% training data
      - Evaluate on 20% validation data
   
   c. Return validation RMSE
      - Optuna tracks best so far
   
4. Get best parameters
   - From trial with lowest validation RMSE
   
5. Train final model
   - Use best parameters
   - Train on FULL training set
   - Evaluate on test set
```

---

## 📊 Results Organization

### Directory Structure:

```
results/
├── EnergyEfficiency/
│   ├── baselines/
│   │   └── models/
│   │       ├── linear_regression.joblib
│   │       └── random_forest.joblib
│   ├── xgboost/
│   │   └── models/
│   │       └── xgboost.joblib
│   ├── tabnet/
│   │   └── models/
│   │       └── tabnet.zip
│   ├── tuning/
│   │   └── optuna/
│   │       └── optuna_results.json
│   ├── shap_plots/
│   │   ├── shap_summary_EnergyEfficiency_xgboost.png
│   │   ├── shap_bar_EnergyEfficiency_xgboost.png
│   │   ├── shap_summary_EnergyEfficiency_randomforest.png
│   │   └── shap_bar_EnergyEfficiency_randomforest.png
│   └── metrics/
│       └── all_metrics.json
│
├── EnergyAppliances/
│   └── [same structure]
│
└── summary.json  # Combined results from both datasets
```

### Metrics JSON Structure:

```json
{
  "EnergyEfficiency": {
    "LinearRegression": {
      "model": "Linear Regression",
      "RMSE": 1.0805,
      "MAE": 0.8166,
      "R2": 0.9888
    },
    "RandomForest": {...},
    "XGBoost": {...},
    "TabNet": {...},
    "TunedXGBoost": {
      "model": "Tuned XGBoost",
      "RMSE": 0.7095,
      "best_params": {...}
    }
  },
  "EnergyAppliances": {...}
}
```

---

## 🔑 Key Concepts Explained

### 1. **Train/Test Split (80/20)**
- **Training Set**: Used to learn patterns
- **Test Set**: Used to evaluate performance (unseen data)
- **Why split?**: Prevents overfitting, gives realistic performance estimate

### 2. **Preprocessing Pipeline**
- **StandardScaler**: Converts numeric features to mean=0, std=1
  - Formula: `(x - mean) / std`
  - Why? Different scales (e.g., temperature vs. area) can bias models
  
- **OneHotEncoder**: Converts categories to binary features
  - Example: `Orientation: [2, 3, 4]` → `Orientation_2: [1,0,0], Orientation_3: [0,1,0], Orientation_4: [0,0,1]`
  - Why? Models need numeric inputs

### 3. **Early Stopping**
- Monitors validation performance during training
- Stops if no improvement for N epochs
- Prevents overfitting
- Uses best weights (from best epoch)

### 4. **Cross-Validation in Optuna**
- Splits training data into train/validation
- Validation set used to evaluate hyperparameters
- Test set only used for final evaluation
- Prevents "cheating" (tuning on test set)

### 5. **SHAP Values**
- **Positive SHAP**: Feature increases prediction
- **Negative SHAP**: Feature decreases prediction
- **Large absolute value**: Feature is important
- **Sum of SHAP values**: Equals prediction - baseline

---

## 🎓 Summary

The pipeline is designed to be:
- **Modular**: Each component can be used independently
- **Reproducible**: Fixed random seeds
- **Robust**: Handles edge cases (missing values, high cardinality)
- **Complete**: Covers all CP322 requirements
- **Well-organized**: Results saved in structured directories

Each model type serves a different purpose:
- **Linear Regression**: Simple baseline, fast, interpretable
- **Random Forest**: Robust, handles non-linearity
- **XGBoost**: Powerful, handles complex patterns
- **TabNet**: Deep learning, learns feature interactions
- **Tuned XGBoost**: Optimized version of XGBoost

The pipeline automatically handles both datasets, trains all models, tunes hyperparameters, generates SHAP plots, and saves everything in an organized structure.

