## Alternative ML Models - Installation & Usage

### Installation

Install additional model libraries:

```bash
# Activate your virtual environment first
source delivery_prediction_venv/bin/activate

# Install LightGBM (recommended - often best for tabular data)
pip install lightgbm

# Install CatBoost (excellent with categorical features)
pip install catboost

# Random Forest and Ridge are already included in scikit-learn
```

### Quick Usage

Train with a specific model:

```bash
# LightGBM (recommended - fast and accurate)
python train_multi_model.py --model lightgbm

# Random Forest (robust, good baseline)
python train_multi_model.py --model random_forest

# CatBoost (best for categorical features)
python train_multi_model.py --model catboost

# Ridge Regression (linear model, fast)
python train_multi_model.py --model ridge

# Gradient Boosting (sklearn implementation)
python train_multi_model.py --model gradient_boosting

# Train ALL models and compare results
python train_multi_model.py --model all
```

### Model Comparison

Run all models to find the best one:

```bash
python train_multi_model.py --model all
```

This will:
1. Train all 5 model types
2. Show metrics for each (R², MAE, RMSE)
3. Rank models by performance
4. Recommend the best model

Expected output:
```
================================================================================
MODEL COMPARISON RESULTS
================================================================================
Model                        R²        MAE       RMSE
--------------------------------------------------------------------------------
lightgbm              0.650      8.45      11.23
random_forest         0.612      9.12      12.45
catboost              0.638      8.67      11.56
gradient_boosting     0.589      9.45      12.89
ridge                 0.423     11.23      15.67

🏆 Best Model: LIGHTGBM
   R²: 0.650
   MAE: 8.45 minutes
```

### Why XGBoost May Be Failing

With R² of -0.021, possible causes:
1. **Insufficient training data** - Need more historical orders
2. **Poor feature quality** - Features not predictive
3. **Data leakage issues** - Point-in-time correctness problems
4. **Hyperparameter issues** - Model not tuned correctly
5. **Data distribution** - Target variable may need transformation

### Model Recommendations

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **LightGBM** | Most cases | Fast, accurate, handles sparse data | Requires tuning |
| **Random Forest** | Noisy data | Robust, hard to overfit | Slower, larger files |
| **CatBoost** | Categorical features | Auto-handles categories | Slower training |
| **Ridge** | Linear relationships | Fast, interpretable | Limited non-linearity |
| **Gradient Boosting** | General use | Solid baseline | Slower than LightGBM |

### Programmatic Usage

```python
from delivery_ml.training.multi_model_pipeline import train_model

# Train LightGBM model
run_id = train_model(
    model_type="lightgbm",
    materialize_features=True,
)

# Train Random Forest
run_id = train_model(
    model_type="random_forest",
    n_estimators=300,  # Custom parameter
)

# Train CatBoost
run_id = train_model(
    model_type="catboost",
    iterations=1000,
)
```

### Model-Specific Parameters

**LightGBM:**
```python
train_model(
    model_type="lightgbm",
    n_estimators=500,
    max_depth=15,
    learning_rate=0.05,
    num_leaves=50,
)
```

**Random Forest:**
```python
train_model(
    model_type="random_forest",
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
)
```

**CatBoost:**
```python
train_model(
    model_type="catboost",
    iterations=500,
    depth=10,
    learning_rate=0.05,
)
```

### Using Trained Models

Models are saved with their type in the filename:
- `delivery_time_model_lightgbm_latest.pkl`
- `delivery_time_model_random_forest_latest.pkl`
- `delivery_time_model_catboost_latest.pkl`
- etc.

Load a specific model:
```python
from delivery_ml.training.multi_model_pipeline import MultiModelTrainer

# Load LightGBM model
model = MultiModelTrainer.load("models/delivery_time_model_lightgbm_latest.pkl")

# Make prediction
prediction = model.predict({
    "haversine_distance": 5.2,
    "hour_of_day": 18,
    "day_of_week": 5,
    # ... other features
})
```

### Troubleshooting

**ImportError for lightgbm/catboost:**
```bash
pip install lightgbm catboost
```

**Still getting poor R²:**
1. Check data quality:
   ```python
   from delivery_ml.features.store import FeatureStore
   store = FeatureStore()
   df = store.offline.get_training_data(train_start, train_end)
   print(df.describe())  # Check for anomalies
   ```

2. Check for data leakage - ensure point-in-time correctness

3. Try feature engineering - add more predictive features

4. Check target distribution:
   ```python
   import polars as pl
   print(df.select("delivery_time_minutes").describe())
   # Consider log transformation if highly skewed
   ```
