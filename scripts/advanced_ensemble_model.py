import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models=None, meta_model=None, cv_folds=5):
        self.base_models = base_models or self._get_default_base_models()
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.cv_folds = cv_folds
        self.trained_base_models = []
        self.scaler = StandardScaler()
        
    def _get_default_base_models(self):
        """Default base models"""
        return {
            'xgb': xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.01,
                random_state=42
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                alpha=0.01
            )
        }
    
    def fit(self, X, y):
        """Fitting the stacking ensemble"""
        print("Training stacking ensemble...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # K-fold cross-validation for generating meta-features
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            print(f"Training fold {fold + 1}/{self.cv_folds}...")
            
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            
            fold_models = {}
            for name, model in self.base_models.items():
                # Clone and fit model
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_train_fold, y_train_fold)
                fold_models[name] = fold_model
                
                # Generate predictions for validation set
                val_pred = fold_model.predict(X_val_fold)
                meta_features[val_idx, list(self.base_models.keys()).index(name)] = val_pred
            
            if fold == 0:  # Store models from first fold for final training
                self.trained_base_models = fold_models
        
        # Train final base models on full dataset
        print("Training final base models on full dataset...")
        final_base_models = {}
        for name, model in self.base_models.items():
            final_model = model.__class__(**model.get_params())
            final_model.fit(X_scaled, y)
            final_base_models[name] = final_model
        
        self.trained_base_models = final_base_models
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Making predictions using the ensemble"""
        X_scaled = self.scaler.transform(X)
        
        # Generate base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.trained_base_models.items()):
            base_predictions[:, i] = model.predict(X_scaled)
        
        # Generate final prediction using meta-model
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions
    
    def get_feature_importance(self, X):
        """Getting ensemble feature importance"""
        importances = {}
        
        for name, model in self.trained_base_models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances[name] = np.abs(model.coef_)
        
        return importances

class TwoStageEnsemble:
    def __init__(self):
        self.churn_model = None
        self.clv_model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, churn_labels=None):
        """Fitting two-stage model: Churn Prediction + CLV regression"""
        print("Training two-stage ensemble model...")
        
        # If churn labels not provided, create them based on recency
        if churn_labels is None:
            churn_threshold = X['Recency'].quantile(0.8)  # Top 20% recency as churned
            churn_labels = (X['Recency'] > churn_threshold).astype(int)
        
        # Stage 1: Train churn prediction model
        print("Stage 1: Training churn prediction model...")
        from sklearn.ensemble import RandomForestClassifier
        self.churn_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.churn_model.fit(X, churn_labels)
        
        # Stage 2: Train CLV model on non-churned customers
        print("Stage 2: Training CLV regression model...")
        non_churned_mask = churn_labels == 0
        X_clv = X[non_churned_mask]
        y_clv = y[non_churned_mask] if hasattr(y, '__getitem__') else y.iloc[non_churned_mask]
        
        self.clv_model = AdvancedStackingRegressor()
        self.clv_model.fit(X_clv, y_clv)
        
        return self
    
    def predict(self, X):
        """Predicting CLV using two-stage approach"""
        # Predict churn probability
        churn_prob = self.churn_model.predict_proba(X)[:, 0]  # Probability of not churning
        
        # Predict CLV for all customers
        clv_pred = self.clv_model.predict(X)
        
        # Combine: CLV = P(not churn) * Predicted CLV
        final_clv = churn_prob * clv_pred
        
        return final_clv

def train_ultimate_ensemble(X, y):
    """Training the ultimate ensemble model"""
    print("Training Ultimate CLV Ensemble Model...")
    
    # Split features into different types for specialized preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create multiple ensemble models
    models = {
        'stacking_ensemble': AdvancedStackingRegressor(),
        'two_stage_ensemble': TwoStageEnsemble(),
        'xgb_optimized': xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.005,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
    }
    
    # Train all models
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X, y)
        trained_models[name] = model
    
    return trained_models

# Usage example
if __name__ == "__main__":
    # Load enhanced features and probabilistic predictions
    features_df = pd.read_csv('data/processed/enhanced_customer_features.csv')
    prob_preds_df = pd.read_csv('outputs/reports/probabilistic_clv_predictions.csv')

    # Merge dataframes
    merged_df = pd.merge(features_df, prob_preds_df[['predicted_clv']], left_index=True, right_index=True)

    # Prepare target variable
    target_clv = merged_df['predicted_clv']
    
    # Select features for modeling
    feature_columns = [col for col in merged_df.columns 
                      if col not in ['CustomerID', 'predicted_clv'] and merged_df[col].dtype in ['int64', 'float64']]
    X = merged_df[feature_columns]
    y = target_clv
    
    # Train ensemble models
    trained_models = train_ultimate_ensemble(X, y)
    
    # Save models
    for name, model in trained_models.items():
        joblib.dump(model, f'outputs/models/{name}_clv_model.pkl')
    
    print("Ultimate ensemble models trained and saved!")
