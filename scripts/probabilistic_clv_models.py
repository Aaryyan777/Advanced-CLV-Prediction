import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter, ModifiedBetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ProbabilisticCLVModels:
    def __init__(self):
        self.bgf = BetaGeoFitter(penalizer_coef=0.0)
        self.ggf = GammaGammaFitter(penalizer_coef=0.0)
        self.mbgf = ModifiedBetaGeoFitter()
        self.summary_data = None
        
    def prepare_data(self, df, customer_id_col='CustomerID', 
                    datetime_col='InvoiceDate', monetary_col='TotalAmount'):
        """This prepares data for probabilistic models"""
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Create summary data for lifetimes
        self.summary_data = summary_data_from_transaction_data(
            df, 
            customer_id_col=customer_id_col,
            datetime_col=datetime_col,
            monetary_value_col=monetary_col,
            observation_period_end=df[datetime_col].max()
        )
        
        # Remove customers with zero frequency
        self.summary_data = self.summary_data[self.summary_data['frequency'] > 0]
        
        return self.summary_data
    
    def fit_bg_nbd_model(self):
        """This fits BG/NBD model for purchase frequency prediction"""
        print("Fitting BG/NBD model...")
        self.bgf.fit(
            self.summary_data['frequency'], 
            self.summary_data['recency'], 
            self.summary_data['T']
        )
        
        # Model diagnostics
        print("BG/NBD Model Parameters:")
        print(f"r: {self.bgf.params_['r']:.4f}")
        print(f"alpha: {self.bgf.params_['alpha']:.4f}")
        print(f"a: {self.bgf.params_['a']:.4f}")
        print(f"b: {self.bgf.params_['b']:.4f}")
        
        return self.bgf
    
    def fit_gamma_gamma_model(self):
        """This fits Gamma-Gamma model for monetary value prediction"""
        print("Fitting Gamma-Gamma model...")
        
        # Filter customers with frequency > 0 for monetary value modeling
        monetary_data = self.summary_data[self.summary_data['monetary_value'] > 0]
        
        self.ggf.fit(
            monetary_data['frequency'],
            monetary_data['monetary_value']
        )
        
        print("Gamma-Gamma Model Parameters:")
        print(f"p: {self.ggf.params_['p']:.4f}")
        print(f"q: {self.ggf.params_['q']:.4f}")
        print(f"v: {self.ggf.params_['v']:.4f}")
        
        return self.ggf
    
    def predict_clv(self, time_periods=12):
        """This predicts Customer Lifetime Value"""
        print(f"Predicting CLV for {time_periods} periods...")
        
        # Predict future purchases
        predicted_purchases = self.bgf.conditional_expected_number_of_purchases_up_to_time(
            time_periods,
            self.summary_data['frequency'],
            self.summary_data['recency'],
            self.summary_data['T']
        )
        
        # Predict average order value
        predicted_avg_order_value = self.ggf.conditional_expected_average_profit(
            self.summary_data['frequency'],
            self.summary_data['monetary_value']
        )
        
        # Calculate CLV
        predicted_clv = predicted_purchases * predicted_avg_order_value
        
        # Add to summary data
        self.summary_data['predicted_purchases'] = predicted_purchases
        self.summary_data['predicted_avg_order_value'] = predicted_avg_order_value
        self.summary_data['predicted_clv'] = predicted_clv
        
        return predicted_clv
    
    def calculate_actual_clv(self, df, periods=12):
        """Calculate actual CLV for validation"""
        # This is a simplified version - you'd need future data for proper validation
        df_sorted = df.sort_values('InvoiceDate')
        cutoff_date = df['InvoiceDate'].quantile(0.7)  # Use 70% for training
        
        train_data = df[df['InvoiceDate'] <= cutoff_date]
        test_data = df[df['InvoiceDate'] > cutoff_date]
        
        # Calculate actual CLV from test period
        actual_clv = test_data.groupby('CustomerID')['TotalAmount'].sum()
        
        return actual_clv
    
    def evaluate_model(self, actual_clv):
        """Evaluate model performance"""
        # Align predictions with actual values
        common_customers = self.summary_data.index.intersection(actual_clv.index)
        
        y_true = actual_clv.loc[common_customers]
        y_pred = self.summary_data.loc[common_customers, 'predicted_clv']
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"Probabilistic Model Performance:")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²: {r2:.4f}")
        
        return {'rmse': rmse, 'r2': r2}
    
    def fit_modified_beta_geo(self):
        """Fit Modified BG model (alternative)"""
        print("Fitting Modified BG model...")
        self.mbgf.fit(
            self.summary_data['frequency'],
            self.summary_data['recency'], 
            self.summary_data['T']
        )
        return self.mbgf

# Enhanced BG/NBD + Gamma-Gamma Implementation
class EnhancedProbabilisticCLV(ProbabilisticCLVModels):
    def __init__(self):
        super().__init__()
        self.regularization_params = {
            'bgf_penalizer': 0.01,
            'ggf_penalizer': 0.01
        }
    
    def fit_enhanced_models(self, penalizer_coef=0.01):
        """Fit models with regularization"""
        print("Fitting enhanced probabilistic models with regularization...")
        
        # Enhanced BG/NBD with regularization
        self.bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        self.bgf.fit(
            self.summary_data['frequency'], 
            self.summary_data['recency'], 
            self.summary_data['T']
        )
        
        # Enhanced Gamma-Gamma with regularization
        monetary_data = self.summary_data[self.summary_data['monetary_value'] > 0]
        self.ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
        self.ggf.fit(
            monetary_data['frequency'],
            monetary_data['monetary_value']
        )
        
        return self.bgf, self.ggf
    
    def predict_clv_with_confidence(self, time_periods=12, confidence_level=0.95):
        """Predict CLV with confidence intervals"""
        # Basic predictions
        predicted_clv = self.predict_clv(time_periods)
        
        # Add confidence intervals (simplified approach)
        std_error = predicted_clv.std()
        margin_error = 1.96 * std_error  # 95% confidence
        
        self.summary_data['clv_lower_bound'] = predicted_clv - margin_error
        self.summary_data['clv_upper_bound'] = predicted_clv + margin_error
        
        return predicted_clv

# Usage example
if __name__ == "__main__":
    # Load your enhanced features
    df = pd.read_csv('data/raw/online_retail_II.csv', encoding='latin1')
    df.rename(columns={'Customer ID': 'CustomerID', 'Invoice': 'InvoiceNo', 'Price': 'UnitPrice'}, inplace=True)
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Initialize probabilistic CLV model
    prob_clv = EnhancedProbabilisticCLV()
    
    # Prepare data
    summary_data = prob_clv.prepare_data(df)
    
    # Fit models
    bgf, ggf = prob_clv.fit_enhanced_models(penalizer_coef=0.01)
    
    # Predict CLV
    predicted_clv = prob_clv.predict_clv_with_confidence(time_periods=12)
    
    # Save results
    prob_clv.summary_data.to_csv('outputs/reports/probabilistic_clv_predictions.csv')
    print("Probabilistic CLV predictions saved!")
