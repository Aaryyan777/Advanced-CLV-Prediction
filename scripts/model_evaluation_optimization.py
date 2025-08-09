import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_ensemble_model import AdvancedStackingRegressor, TwoStageEnsemble

class AdvancedModelEvaluator:
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def time_series_validation(self, models, X, y, n_splits=5):
        """Time-based cross-validation for CLV models"""
        print("Performing time-series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            r2_scores = []
            rmse_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                r2_scores.append(r2)
                rmse_scores.append(rmse)
            
            results[name] = {
                'R2_mean': np.mean(r2_scores),
                'R2_std': np.std(r2_scores),
                'RMSE_mean': np.mean(rmse_scores),
                'RMSE_std': np.std(rmse_scores)
            }
            
            print(f"{name} - R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
            print(f"{name} - RMSE: {np.mean(rmse_scores):,.2f} (±{np.std(rmse_scores):.2f})")
        
        self.results = results
        return results
    
    def bayesian_optimization(self, model_class, X, y, param_space, n_trials=100):
        """Bayesian optimization for hyperparameter tuning"""
        print(f"Running Bayesian optimization with {n_trials} trials...")
        
        def objective(trial):
            # Define hyperparameters based on model type
            if 'XGB' in str(model_class):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
            else:
                params = param_space
            
            # Create and evaluate model
            model = model_class(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best R² score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def feature_importance_analysis(self, model, X, feature_names=None):
        """Analyze feature importance"""
        if feature_names is None:
            feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("Model doesn't support feature importance analysis")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def residual_analysis(self, model, X, y):
        """Perform residual analysis"""
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Calculate metrics
        metrics = {
            'R2': r2_score(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'Mean_Residual': np.mean(residuals),
            'Std_Residual': np.std(residuals)
        }
        
        return metrics, residuals
    
    def ensemble_weight_optimization(self, models, X, y):
        """Optimize ensemble weights"""
        print("Optimizing ensemble weights...")
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X)
        
        pred_matrix = np.column_stack(list(predictions.values()))
        model_names = list(predictions.keys())
        
        def objective(trial):
            # Generate weights that sum to 1
            weights = []
            for i in range(len(model_names)):
                if i == len(model_names) - 1:
                    weights.append(1.0 - sum(weights))
                else:
                    weights.append(trial.suggest_float(f'weight_{i}', 0.0, 1.0 - sum(weights)))
            
            # Weighted ensemble prediction
            ensemble_pred = np.dot(pred_matrix, weights)
            
            # Return R² score
            return r2_score(y, ensemble_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1000)
        
        optimal_weights = list(study.best_params.values())
        optimal_weights.append(1.0 - sum(optimal_weights))
        
        return dict(zip(model_names, optimal_weights)), study.best_value

def comprehensive_model_evaluation():
    """Complete model evaluation pipeline"""
    # Load data
    features_df = pd.read_csv('data/processed/enhanced_customer_features.csv')
    
    # Prepare features and target
    # Exclude non-numeric columns and the target variable itself
    numeric_features_df = features_df.select_dtypes(include=np.number)
    feature_columns = [col for col in numeric_features_df.columns 
                      if col not in ['CustomerID', 'Monetary']]
    X = numeric_features_df[feature_columns]
    y = numeric_features_df['Monetary']  # Assuming 'Monetary' is numeric and the target
    
    # Load trained models
    models = {}
    model_files = ['outputs/models/stacking_ensemble_clv_model.pkl', 
                   'outputs/models/two_stage_ensemble_clv_model.pkl',
                   'outputs/models/xgb_optimized_clv_model.pkl']
    
    for file in model_files:
        try:
            model_name = file.replace('_clv_model.pkl', '')
            models[model_name] = joblib.load(file)
        except FileNotFoundError:
            print(f"Model file {file} not found. Skipping...")
    
    # Initialize evaluator
    evaluator = AdvancedModelEvaluator()
    
    # Perform comprehensive evaluation
    print("=== COMPREHENSIVE MODEL EVALUATION ===")
    
    # Time-series validation
    ts_results = evaluator.time_series_validation(models, X, y)
    
    # Find best model
    best_model_name = max(ts_results, key=lambda x: ts_results[x]['R2_mean'])
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best R² Score: {ts_results[best_model_name]['R2_mean']:.4f}")
    
    # Feature importance analysis
    if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
        importance_df = evaluator.feature_importance_analysis(best_model, X)
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10))
    
    # Residual analysis
    metrics, residuals = evaluator.residual_analysis(best_model, X, y)
    print(f"\nResidual Analysis:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    results_summary = {
        'model_comparison': ts_results,
        'best_model': best_model_name,
        'best_metrics': metrics,
        'feature_importance': importance_df.to_dict() if 'importance_df' in locals() else None
    }
    
    import json
    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4, default=str)
    
    print("\nEvaluation complete! Results saved to 'model_evaluation_results.json'")
    
    return evaluator, best_model

if __name__ == "__main__":
    evaluator, best_model = comprehensive_model_evaluation()
