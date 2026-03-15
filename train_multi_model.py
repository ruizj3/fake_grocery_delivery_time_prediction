#!/usr/bin/env python3
"""Train delivery time prediction model with multiple model options.

Usage:
    python train_multi_model.py --model lightgbm
    python train_multi_model.py --model random_forest
    python train_multi_model.py --model catboost
    python train_multi_model.py --model ridge
    python train_multi_model.py --model gradient_boosting
    python train_multi_model.py --model extra_trees
    python train_multi_model.py --model all  # Train all models and compare

    # With hyperparameter tuning (RandomizedSearchCV)
    python train_multi_model.py --model lightgbm --tune
    python train_multi_model.py --model all --tune

    # Customize tuning
    python train_multi_model.py --model lightgbm --tune --search-type grid --cv 3
    python train_multi_model.py --model random_forest --tune --n-iter 50
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from delivery_ml.training.multi_model_pipeline import train_model, ModelType
from delivery_ml.features.store import FeatureStore


def compare_all_models(
    output_file: str = "model_comparison_results.txt",
    tune: bool = False,
    tune_kwargs: dict | None = None,
):
    """Train all available models and compare results."""
    models: list[ModelType] = ["lightgbm", "random_forest", "catboost", "ridge", "gradient_boosting", "extra_trees"]
    
    results = []
    output_lines = []
    
    header = "=" * 80
    title = "TRAINING ALL MODELS FOR COMPARISON"
    
    print(header)
    print(title)
    print(header)
    
    output_lines.append(header)
    output_lines.append(title)
    output_lines.append(header)
    
    for model_type in models:
        try:
            msg = f"\n{'=' * 80}\nTraining {model_type.upper()} model...\n{'=' * 80}\n"
            print(msg)
            output_lines.append(msg)
            
            run_id = train_model(
                model_type=model_type,
                materialize_features=(model_type == models[0]),  # Only materialize once
                tune=tune,
                tune_kwargs=tune_kwargs,
            )
            
            # Load metrics from saved model
            from delivery_ml.config import settings
            model_path = settings.model_dir / f"delivery_time_model_{model_type}_latest.pkl"
            
            if model_path.exists():
                from delivery_ml.training.multi_model_pipeline import MultiModelTrainer
                loaded_model = MultiModelTrainer.load(model_path)
                results.append({
                    "model": model_type,
                    "r2": loaded_model.metrics.get("r2", 0),
                    "mae": loaded_model.metrics.get("mae", 0),
                    "rmse": loaded_model.metrics.get("rmse", 0),
                    "train_r2": loaded_model.metrics.get("train_r2", 0),
                    "train_mae": loaded_model.metrics.get("train_mae", 0),
                })
            
        except ImportError as e:
            error_msg = f"⚠️  Skipping {model_type}: {e}"
            print(error_msg)
            output_lines.append(error_msg)
        except Exception as e:
            error_msg = f"❌ Error training {model_type}: {e}"
            print(error_msg)
            output_lines.append(error_msg)
    
    # Print comparison
    if results:
        output_lines.append("\n" + "=" * 80)
        output_lines.append("MODEL COMPARISON RESULTS")
        output_lines.append("=" * 80)
        output_lines.append(f"{'Model':<20} {'Test R²':>10} {'Test MAE':>10} {'Test RMSE':>10} {'Train R²':>10}")
        output_lines.append("-" * 80)
        
        # Sort by R² descending
        results.sort(key=lambda x: x["r2"], reverse=True)
        
        for result in results:
            line = f"{result['model']:<20} {result['r2']:>10.3f} {result['mae']:>10.2f} {result['rmse']:>10.2f} {result['train_r2']:>10.3f}"
            output_lines.append(line)
        
        output_lines.append("")
        output_lines.append("🏆 Best Model: " + results[0]["model"].upper())
        output_lines.append(f"   Test R²: {results[0]['r2']:.3f}")
        output_lines.append(f"   Test MAE: {results[0]['mae']:.2f} minutes")
        output_lines.append(f"   Train R²: {results[0]['train_r2']:.3f}")
        
        # Print to console
        for line in output_lines[-8:]:  # Print summary section
            print(line)
        
        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"model_comparison_{timestamp}.txt"
        
        with open(output_filename, "w") as f:
            f.write("\n".join(output_lines))
            f.write(f"\n\nGenerated at: {datetime.now().isoformat()}\n")
        
        print(f"\n✅ Results saved to: {output_filename}")



def main():
    parser = argparse.ArgumentParser(description="Train delivery time prediction model")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "random_forest", "catboost", "ridge", "gradient_boosting", "extra_trees", "all"],
        help="Model type to train (default: lightgbm)",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Skip feature materialization (faster but may reduce accuracy)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning before training",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="random",
        choices=["random", "grid"],
        help="Tuning search strategy: 'random' (fast, default) or 'grid' (exhaustive)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds for tuning (default: 5)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of parameter combinations for RandomizedSearchCV (default: 30)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="neg_mean_absolute_error",
        help="Scoring metric for tuning (default: neg_mean_absolute_error)",
    )
    
    args = parser.parse_args()
    
    # Build tune_kwargs from CLI args
    tune_kwargs = None
    if args.tune:
        tune_kwargs = {
            "search_type": args.search_type,
            "cv": args.cv,
            "n_iter": args.n_iter,
            "scoring": args.scoring,
        }
    
    if args.model == "all":
        compare_all_models(tune=args.tune, tune_kwargs=tune_kwargs)
    else:
        print(f"Training {args.model.upper()} model...")
        run_id = train_model(
            model_type=args.model,  # type: ignore
            materialize_features=not args.no_materialize,
            tune=args.tune,
            tune_kwargs=tune_kwargs,
        )
        
        # Save single model metrics to file
        from delivery_ml.config import settings
        model_path = settings.model_dir / f"delivery_time_model_{args.model}_latest.pkl"
        
        if model_path.exists():
            from delivery_ml.training.multi_model_pipeline import MultiModelTrainer
            loaded_model = MultiModelTrainer.load(model_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"model_{args.model}_{timestamp}.txt"
            
            with open(output_filename, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"{args.model.upper()} MODEL TRAINING RESULTS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Model Type: {args.model}\n")
                f.write(f"MLflow Run ID: {run_id}\n")
                f.write(f"Model Path: {model_path}\n\n")
                f.write("Test Metrics:\n")
                f.write(f"  R²: {loaded_model.metrics.get('r2', 0):.3f}\n")
                f.write(f"  MAE: {loaded_model.metrics.get('mae', 0):.2f} minutes\n")
                f.write(f"  RMSE: {loaded_model.metrics.get('rmse', 0):.2f} minutes\n\n")
                f.write("Training Metrics:\n")
                f.write(f"  R²: {loaded_model.metrics.get('train_r2', 0):.3f}\n")
                f.write(f"  MAE: {loaded_model.metrics.get('train_mae', 0):.2f} minutes\n\n")
                f.write(f"Training Size: {loaded_model.metrics.get('train_size', 0)}\n")
                f.write(f"Test Size: {loaded_model.metrics.get('test_size', 0)}\n\n")
                f.write(f"Generated at: {datetime.now().isoformat()}\n")
            
            print(f"\n✅ Results saved to: {output_filename}")


if __name__ == "__main__":
    main()
