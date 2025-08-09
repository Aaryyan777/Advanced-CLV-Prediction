import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from advanced_ensemble_model import TwoStageEnsemble, AdvancedStackingRegressor

class CustomerSegmentation:
    def __init__(self):
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_analysis = None
        
    def find_optimal_clusters(self, clv_predictions, max_clusters=8):
        """Find optimal number of clusters using elbow method and silhouette score"""
        clv_scaled = self.scaler.fit_transform(clv_predictions.reshape(-1, 1))
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(clv_scaled)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(clv_scaled, labels)
            silhouette_scores.append(silhouette_avg)
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.4f}")
        
        return optimal_k
    
    def perform_clustering(self, enhanced_features, clv_predictions, n_clusters=3):
        """Perform K-Means clustering on CLV predictions"""
        print(f"Performing K-Means clustering with {n_clusters} clusters...")
        
        # Scale CLV predictions for clustering
        clv_scaled = self.scaler.fit_transform(clv_predictions.reshape(-1, 1))
        
        # Apply K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(clv_scaled)
        
        # Add cluster labels to features
        enhanced_features = enhanced_features.copy()
        enhanced_features['Predicted_CLV'] = clv_predictions
        enhanced_features['Cluster'] = cluster_labels
        
        return enhanced_features, cluster_labels
    
    def analyze_clusters(self, features_with_clusters):
        """Detailed analysis of each customer cluster"""
        print("Performing detailed cluster analysis...")
        
        # Group by cluster and calculate statistics
        cluster_stats = features_with_clusters.groupby('Cluster').agg({
            'Recency': ['mean', 'count'],
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Predicted_CLV': 'mean',
            'CustomerID': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = ['Avg_Recency', 'Customer_Count', 'Avg_Frequency', 
                               'Avg_Monetary', 'Avg_Predicted_CLV', 'Total_Customers']
        
        # Remove duplicate count column
        cluster_stats = cluster_stats.drop('Customer_Count', axis=1)
        
        # Add percentage of total customers
        total_customers = len(features_with_clusters)
        cluster_stats['Percentage_of_Total'] = (cluster_stats['Total_Customers'] / total_customers * 100).round(1)
        
        # Sort by predicted CLV (descending)
        cluster_stats = cluster_stats.sort_values('Avg_Predicted_CLV', ascending=False)
        
        self.cluster_analysis = cluster_stats
        
        return cluster_stats
    
    def create_cluster_labels(self, cluster_stats):
        """Create meaningful labels for clusters based on CLV"""
        labels = {}
        sorted_clusters = cluster_stats.sort_values('Avg_Predicted_CLV', ascending=False)
        
        if len(sorted_clusters) == 3:
            cluster_indices = sorted_clusters.index.tolist()
            labels[cluster_indices[0]] = "High-Value Customers"
            labels[cluster_indices[1]] = "Medium-Value Customers" 
            labels[cluster_indices[2]] = "Low-Value Customers"
        else:
            # For more clusters, create dynamic labels
            for i, cluster_id in enumerate(sorted_clusters.index):
                if i == 0:
                    labels[cluster_id] = "VIP Champions"
                elif i == 1:
                    labels[cluster_id] = "High-Value Loyalists"
                elif i == len(sorted_clusters) - 1:
                    labels[cluster_id] = "At-Risk Low-Value"
                else:
                    labels[cluster_id] = f"Medium-Value Tier {i}"
        
        return labels
    
    def print_cluster_insights(self, cluster_stats, cluster_labels):
        """Print detailed cluster insights matching your original format"""
        print("\n" + "="*60)
        print("CUSTOMER SEGMENTATION ANALYSIS")
        print("="*60)
        
        for cluster_id in cluster_stats.index:
            stats = cluster_stats.loc[cluster_id]
            label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
            
            print(f"\n{label} (Cluster {cluster_id}):")
            print(f"   Average Recency: {stats['Avg_Recency']:.1f} days")
            print(f"   Average Frequency: {stats['Avg_Frequency']:.0f} purchases")
            print(f"   Average Monetary Value: ${stats['Avg_Monetary']:,.0f}")
            print(f"   Average Predicted CLV: ${stats['Avg_Predicted_CLV']:,.0f}")
            print(f"   Number of Customers: {stats['Total_Customers']:,}")
            print(f"   Percentage of Total: {stats['Percentage_of_Total']:.1f}%")
            
            # Add characteristics based on the values
            if label == "High-Value Customers":
                print("   Characteristics: Most recent, frequent, and highest-spending customers. Critical for business growth.")
            elif label == "Medium-Value Customers":
                print("   Characteristics: Good engagement with significant growth potential. Valuable segment for upselling.")
            elif label == "Low-Value Customers":
                print("   Characteristics: Largest segment with lowest CLV. Likely inactive or at risk of churn.")
    
    def visualize_clusters(self, features_with_clusters, cluster_labels):
        """Create visualizations for cluster analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. CLV Distribution by Cluster
        axes[0, 0].boxplot([features_with_clusters[features_with_clusters['Cluster'] == i]['Predicted_CLV'] 
                           for i in sorted(features_with_clusters['Cluster'].unique())])
        axes[0, 0].set_title('CLV Distribution by Cluster')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Predicted CLV ($)')
        
        # 2. Cluster Size Distribution
        cluster_counts = features_with_clusters['Cluster'].value_counts().sort_index()
        axes[0, 1].pie(cluster_counts.values, labels=[cluster_labels.get(i, f'Cluster {i}') for i in cluster_counts.index], 
                      autopct='%1.1f%%')
        axes[0, 1].set_title('Customer Distribution by Cluster')
        
        # 3. RFM Analysis by Cluster
        rfm_means = features_with_clusters.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        rfm_means.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average RFM Values by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].legend()
        
        # 4. Scatter plot: Frequency vs Monetary colored by Cluster
        for cluster_id in sorted(features_with_clusters['Cluster'].unique()):
            cluster_data = features_with_clusters[features_with_clusters['Cluster'] == cluster_id]
            axes[1, 1].scatter(cluster_data['Frequency'], cluster_data['Monetary'], 
                             label=cluster_labels.get(cluster_id, f'Cluster {cluster_id}'), alpha=0.6)
        axes[1, 1].set_title('Customer Segments: Frequency vs Monetary')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Monetary')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

# Load the enhanced features and make predictions
print("Loading enhanced features for segmentation...")
features_df = pd.read_csv('data/processed/enhanced_customer_features.csv')
prob_preds_df = pd.read_csv('outputs/reports/probabilistic_clv_predictions.csv')

# Merge dataframes on CustomerID to align features and predictions
# This ensures that only customers for whom CLV predictions were made are included
merged_df = pd.merge(features_df, prob_preds_df[['CustomerID', 'predicted_clv']], on='CustomerID', how='inner')

# Load the best model (two-stage ensemble from our previous results)
best_model = joblib.load('outputs/models/two_stage_ensemble_clv_model.pkl')

# Prepare features for prediction, mirroring advanced_ensemble_model.py
# Exclude 'CustomerID' and 'predicted_clv' from features, and ensure numeric types
feature_columns = [col for col in merged_df.columns 
                  if col not in ['CustomerID', 'predicted_clv'] and merged_df[col].dtype in ['int64', 'float64']]
X = merged_df[feature_columns]

# Generate CLV predictions using the aligned features
print("Generating CLV predictions for segmentation...")
clv_predictions = best_model.predict(X)

# Add the predicted CLV to the merged dataframe for clustering
merged_df['Predicted_CLV'] = clv_predictions

# Initialize customer segmentation
segmentation = CustomerSegmentation()

# Find optimal number of clusters using the predicted CLV
optimal_k = segmentation.find_optimal_clusters(merged_df['Predicted_CLV'].values, max_clusters=6)

# Perform clustering using the predicted CLV from the merged dataframe
# Use the merged_df for clustering and analysis, as it contains all necessary features and aligned predictions
features_with_clusters, cluster_labels = segmentation.perform_clustering(
    merged_df, merged_df['Predicted_CLV'].values, n_clusters=3
)

# Analyze clusters
cluster_stats = segmentation.analyze_clusters(features_with_clusters)

# Create meaningful cluster labels
cluster_label_names = segmentation.create_cluster_labels(cluster_stats)

# Print detailed insights
segmentation.print_cluster_insights(cluster_stats, cluster_label_names)

# Create visualizations
segmentation.visualize_clusters(features_with_clusters, cluster_label_names)

# Save segmentation results
features_with_clusters.to_csv('data/processed/customer_segments.csv', index=False)
cluster_stats.to_csv('outputs/reports/cluster_analysis.csv')

print("Customer segmentation complete!")
print("Results saved:")
print("   • Customer segments: data/processed/customer_segments.csv")
print("   • Cluster analysis: outputs/reports/cluster_analysis.csv")
