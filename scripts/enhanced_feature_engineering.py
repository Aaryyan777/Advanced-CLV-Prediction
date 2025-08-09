import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_time_features(self, df):
        """This creates time-based features"""
        df = df.copy()
        
        # Convert to datetime if not already
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Seasonal features
        df['Month'] = df['InvoiceDate'].dt.month
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Seasonal indicators
        df['IsSummer'] = df['Month'].isin([6, 7, 8]).astype(int)
        df['IsWinter'] = df['Month'].isin([12, 1, 2]).astype(int)
        df['IsHoliday'] = df['Month'].isin([11, 12]).astype(int)  # Holiday season
        
        return df
    
    def create_rfm_plus_features(self, df):
        """Enhanced RFM with additional behavioral metrics"""
        reference_date = df['InvoiceDate'].max() + timedelta(days=1)
        
        # Group by customer
        customer_features = df.groupby('CustomerID').agg({
            'InvoiceDate': ['max', 'min', 'count'],
            'Quantity': ['sum', 'mean', 'std', 'max'],
            'UnitPrice': ['mean', 'std', 'max', 'min'],
            'TotalAmount': ['sum', 'mean', 'std', 'max'],
            'StockCode': ['nunique'],  # Product diversity
            'InvoiceNo': ['nunique'],  # Number of transactions
            'Month': ['nunique'],  # Months active
            'Quarter': ['nunique']   # Quarters active
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['CustomerID'] + [
            f"{col[0]}_{col[1]}" for col in customer_features.columns[1:]
        ]
        
        # Calculate advanced RFM metrics
        customer_features['Recency'] = (reference_date - customer_features['InvoiceDate_max']).dt.days
        customer_features['Frequency'] = customer_features['InvoiceNo_nunique']
        customer_features['Monetary'] = customer_features['TotalAmount_sum']
        
        # Advanced behavioral features
        customer_features['AvgDaysBetweenPurchases'] = (
            (customer_features['InvoiceDate_max'] - customer_features['InvoiceDate_min']).dt.days / 
            customer_features['Frequency'].clip(lower=1)
        )
        
        customer_features['CustomerLifespan'] = (
            customer_features['InvoiceDate_max'] - customer_features['InvoiceDate_min']
        ).dt.days
        
        customer_features['PurchaseVelocity'] = (
            customer_features['Frequency'] / customer_features['CustomerLifespan'].clip(lower=1)
        )
        
        customer_features['AvgOrderValue'] = (
            customer_features['Monetary'] / customer_features['Frequency']
        )
        
        customer_features['ProductDiversity'] = customer_features['StockCode_nunique']
        customer_features['MonthsActive'] = customer_features['Month_nunique']
        customer_features['QuartersActive'] = customer_features['Quarter_nunique']
        
        # Consistency metrics
        customer_features['QuantityConsistency'] = (
            customer_features['Quantity_mean'] / customer_features['Quantity_std'].fillna(1)
        )
        
        customer_features['PriceConsistency'] = (
            customer_features['UnitPrice_mean'] / customer_features['UnitPrice_std'].fillna(1)
        )
        
        # Engagement trend (simplified)
        customer_features['EngagementTrend'] = (
            customer_features['Frequency'] / customer_features['MonthsActive'].clip(lower=1)
        )
        
        return customer_features
    
    def create_polynomial_features(self, df):
        """This createspolynomial and interaction features"""
        df = df.copy()
        
        # Log transformations for skewed features
        log_features = ['Monetary', 'AvgOrderValue', 'TotalAmount_sum', 'TotalAmount_max']
        for feature in log_features:
            if feature in df.columns:
                df[f'Log_{feature}'] = np.log1p(df[feature])
        
        # Polynomial features for key metrics
        df['Recency_Squared'] = df['Recency'] ** 2
        df['Frequency_Squared'] = df['Frequency'] ** 2
        df['Monetary_Squared'] = df['Monetary'] ** 2
        
        # Interaction features
        df['Recency_Frequency'] = df['Recency'] * df['Frequency']
        df['Recency_Monetary'] = df['Recency'] * df['Monetary']
        df['Frequency_Monetary'] = df['Frequency'] * df['Monetary']
        df['RFM_Combined'] = df['Recency'] * df['Frequency'] * df['Monetary']
        
        # Ratios
        df['MonetaryPerFrequency'] = df['Monetary'] / df['Frequency'].clip(lower=1)
        df['FrequencyPerRecency'] = df['Frequency'] / df['Recency'].clip(lower=1)
        
        return df
    
    def create_segmentation_features(self, df):
        """This creates customer segmentation features"""
        df = df.copy()
        
        # RFM scoring (1-5 scale)
        df['R_Score'] = pd.qcut(df['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        df['M_Score'] = pd.qcut(df['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        df['R_Score'] = pd.to_numeric(df['R_Score'])
        df['F_Score'] = pd.to_numeric(df['F_Score'])
        df['M_Score'] = pd.to_numeric(df['M_Score'])
        
        # Combined RFM score
        df['RFM_Score'] = df['R_Score'] * 100 + df['F_Score'] * 10 + df['M_Score']
        
        # Customer value categories
        df['HighValue'] = (df['M_Score'] >= 4).astype(int)
        df['HighFrequency'] = (df['F_Score'] >= 4).astype(int)
        df['Recent'] = (df['R_Score'] >= 4).astype(int)
        
        return df
    
    def fit_transform(self, df):
        """This is the complete feature engineering pipeline"""
        print("Creating time-based features...")
        df = self.create_time_features(df)
        
        print("Creating enhanced RFM features...")
        customer_features = self.create_rfm_plus_features(df)
        
        print("Creating polynomial features...")
        customer_features = self.create_polynomial_features(customer_features)
        
        print("Creating segmentation features...")
        customer_features = self.create_segmentation_features(customer_features)
        
        # Fill missing values
        customer_features = customer_features.fillna(0)
        
        # Remove infinite values
        customer_features = customer_features.replace([np.inf, -np.inf], 0)
        
        print(f"Final feature set shape: {customer_features.shape}")
        return customer_features

# Usage example
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/raw/online_retail_II.csv', encoding='latin1')
    df.rename(columns={'Customer ID': 'CustomerID', 'Invoice': 'InvoiceNo', 'Price': 'UnitPrice'}, inplace=True)
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Initialize feature engineering
    fe = AdvancedFeatureEngineering()
    
    # Create enhanced features
    enhanced_features = fe.fit_transform(df)
    
    # Save enhanced features
    enhanced_features.to_csv('data/processed/enhanced_customer_features.csv', index=False)
    print("Enhanced features saved successfully!")
