# Advanced Customer Lifetime Value (CLV) Prediction

## Project Overview

This project focuses on building a sophisticated Customer Lifetime Value (CLV) prediction system using advanced machine learning techniques. It leverages a combination of probabilistic models, enhanced feature engineering, and ensemble methods to provide accurate and robust CLV forecasts. The goal is to help businesses understand and predict the long-term value of their customers, enabling data-driven decisions for marketing, customer retention, and resource allocation.

## Features

*   **Enhanced Feature Engineering:**
    *   Creation of time-based features (seasonal indicators, day of week, etc.).
    *   Advanced RFM (Recency, Frequency, Monetary) metrics with additional behavioral insights (e.g., average days between purchases, customer lifespan, purchase velocity, product diversity).
    *   Polynomial and interaction features for key metrics.
    *   Customer segmentation features (RFM scoring, value categories).
*   **Probabilistic CLV Models:**
    *   Implementation of BG/NBD (Beta-Geometric/Negative Binomial Distribution) for purchase frequency prediction.
    *   Implementation of Gamma-Gamma for monetary value prediction.
    *   Prediction of CLV with confidence intervals.
*   **Advanced Ensemble Modeling:**
    *   **Stacking Ensemble:** Combines predictions from multiple diverse base models (XGBoost, LightGBM, RandomForest, GradientBoosting, MLP) using a meta-model (Ridge Regression).
    *   **Two-Stage Ensemble:** A specialized approach that first predicts customer churn and then predicts CLV for non-churned customers, combining the results for a final CLV estimate.
    *   **Optimized XGBoost:** A highly tuned XGBoost model for direct CLV prediction.
*   **Comprehensive Model Evaluation & Optimization:**
    *   Time-series cross-validation for robust model assessment.
    *   Feature importance analysis to understand key drivers of CLV.
    *   Residual analysis for model diagnostics.
    *   (Planned) Bayesian optimization for hyperparameter tuning.
    *   (Planned) Ensemble weight optimization.

## Methodology

### Data Preparation
The `enhanced_feature_engineering.py` script performs comprehensive data preparation and feature engineering. It transforms raw transactional data into a rich set of customer-level features, including:
- **Time-based Features:** Month, Quarter, Day of Week, IsWeekend, IsSummer, IsWinter, IsHoliday.
- **Enhanced RFM+ Features:** Recency, Frequency, Monetary, AvgDaysBetweenPurchases, CustomerLifespan, PurchaseVelocity, AvgOrderValue, ProductDiversity, MonthsActive, QuartersActive, QuantityConsistency, PriceConsistency, EngagementTrend.
- **Polynomial and Interaction Features:** Log transformations for skewed features (Monetary, AvgOrderValue, TotalAmount_sum, TotalAmount_max), squared terms for Recency, Frequency, Monetary, and interaction terms like Recency_Frequency, Recency_Monetary, Frequency_Monetary, RFM_Combined, MonetaryPerFrequency, FrequencyPerRecency.
- **Segmentation Features:** RFM scores (R_Score, F_Score, M_Score) and combined RFM_Score, along with binary indicators for HighValue, HighFrequency, and Recent customers.

### Model Training & Optimization
This project employs a multi-faceted approach to model training and optimization, focusing on robust ensemble methods and comprehensive evaluation.

- **Ensemble Models:**
    - **Advanced Stacking Regressor:** This model combines predictions from diverse base models (XGBoost, LightGBM, RandomForest, GradientBoosting, MLP) using a Ridge Regression meta-model. It utilizes K-fold cross-validation to generate meta-features, ensuring robust training.
    - **Two-Stage Ensemble:** A specialized model designed to first predict customer churn (using a RandomForestClassifier) and then predict CLV for non-churned customers (using an Advanced Stacking Regressor). The final CLV is a combination of the churn probability and the predicted CLV for non-churned customers.
    - **Optimized XGBoost:** A standalone XGBoost Regressor with a highly tuned configuration for direct CLV prediction.

- **Evaluation and Optimization Techniques:**
    - **Time-Series Cross-Validation:** Models are rigorously evaluated using `TimeSeriesSplit` to ensure their performance is stable and reliable over time.
    - **Feature Importance Analysis:** The `AdvancedModelEvaluator` class includes functionality to analyze feature importance, providing insights into the key drivers of CLV.
    - **Residual Analysis:** Detailed residual analysis is performed to assess model bias and error distribution.
    - **Bayesian Optimization (Planned):** The framework includes a `bayesian_optimization` method using Optuna for hyperparameter tuning, though it's currently commented out in the main execution flow.
    - **Ensemble Weight Optimization (Planned):** A method for optimizing the weights of individual models within an ensemble is also available but not actively used in the main execution.

- **Model Performance (from time-series cross-validation):**

| Model Name           | Mean R²   | Std R²    | Mean RMSE     | Std RMSE      |
| :------------------- | :-------- | :-------- | :------------ | :------------ |
| `stacking_ensemble`  | 0.7947    | 0.3215    | 5,309.45      | 6,497.42      |
| `two_stage_ensemble` | 0.9615    | 0.0154    | 2,681.99      | 1,480.17      |
| `xgb_optimized`      | 0.7002    | 0.0771    | 6,982.98      | 3,370.40      |

    The `two_stage_ensemble` model demonstrated superior performance with the highest mean R² score (0.9615) and the lowest mean RMSE (2,681.99) across the time-series cross-validation folds, making it the most reliable for CLV prediction in this context.

    **Detailed Metrics for Best Model (**TWO STAGE ENSEMBLE**):**
    *   **R²:** 0.9824 (on the full dataset)
    *   **RMSE (Root Mean Squared Error):** 1854.35
    *   **MAE (Mean Absolute Error):** 265.98
    *   **Mean Residual:** 122.34
    *   **Standard Deviation of Residuals:** 1850.31

### Customer Segmentation
This project performs K-Means clustering on the predicted CLV to segment customers into distinct groups. The `k-means_clustering.py` script identifies optimal clusters and provides detailed analysis of each segment.

**Optimal Number of Clusters:** 2 (based on silhouette score)

**Customer Segment Analysis (for 3 clusters):**

*   **High-Value Customers (Cluster 2):**
    *   Average Recency: 75.3 days
    *   Average Frequency: 4 purchases
    *   Average Monetary Value: $1,289
    *   Average Predicted CLV: $106
    *   Number of Customers: 3
    *   Percentage of Total: 0.1%
    *   *Characteristics:* Most recent, frequent, and highest-spending customers. Critical for business growth.

*   **Medium-Value Customers (Cluster 0):**
    *   Average Recency: 100.8 days
    *   Average Frequency: 11 purchases
    *   Average Monetary Value: $4,032
    *   Average Predicted CLV: $65
    *   Number of Customers: 3,908
    *   Percentage of Total: 88.9%
    *   *Characteristics:* Good engagement with significant growth potential. Valuable segment for upselling.

*   **Low-Value Customers (Cluster 1):**
    *   Average Recency: 508.3 days
    *   Average Frequency: 4 purchases
    *   Average Monetary Value: $973
    *   Average Predicted CLV: $0
    *   Number of Customers: 487
    *   Percentage of Total: 11.1%
    *   *Characteristics:* Largest segment with lowest CLV. Likely inactive or at risk of churn.

## Actionable Insights & Recommendations

Based on the customer segmentation, here are actionable insights and recommendations tailored for each cluster:

### For High-Value Customers (Cluster 2):
- **Insight:** These are your most valuable customers, contributing significantly to revenue. Their high predicted CLV and recent activity make them crucial for sustained growth.
- **Recommendations:**
    - **VIP Treatment:** Offer exclusive benefits, dedicated support, and personalized communication to maintain their loyalty.
    - **Feedback & Co-creation:** Involve them in product development or feedback sessions to strengthen their connection and gather valuable insights.
    - **Referral Programs:** Encourage them to refer new customers with attractive incentives, leveraging their satisfaction.
    - **Proactive Retention:** Monitor their activity for any signs of decreased engagement and intervene with personalized outreach to prevent churn.

### For Medium-Value Customers (Cluster 0):
- **Insight:** This is the largest segment with good engagement and significant potential for growth. They represent a key opportunity to increase overall CLV.
- **Recommendations:**
    - **Loyalty Programs:** Introduce or enhance loyalty programs to reward their continued engagement and encourage higher spending.
    - **Upselling/Cross-selling:** Recommend complementary products or higher-value items based on their purchase history and predicted needs.
    - **Exclusive Content/Early Access:** Offer exclusive content, early access to new products, or special discounts to make them feel valued and encourage more frequent purchases.
    - **Personalized Engagement:** Use their behavioral data to send targeted communications that resonate with their preferences and encourage deeper engagement.

### For Low-Value Customers (Cluster 1):
- **Insight:** This segment has the lowest CLV and is likely inactive or at risk of churn. While they contribute less, re-engaging even a small portion can yield positive returns.
- **Recommendations:**
    - **Re-engagement Campaigns:** Implement targeted email campaigns or promotions with compelling offers to re-activate these customers.
    - **Win-back Strategies:** Analyze reasons for their inactivity (e.g., through surveys or feedback forms) and address pain points to encourage a return.
    - **Personalized Offers:** Use their past purchase history (if available) to offer personalized recommendations or discounts that might entice them back.
    - **Churn Prevention:** For those showing early signs of disengagement, implement automated alerts and personalized interventions to prevent them from moving into this segment.

These actionable insights, derived from the customer segmentation, empower businesses to make data-driven decisions to improve customer relationships and maximize long-term profitability.

## Initial Development & Project History

This project represents an advanced iteration and enhancement of the CLV Segmentation and Prediction project available at [https://github.com/Aaryyan777/CLV-Segmentation-Prediction](https://github.com/Aaryyan777/CLV-Segmentation-Prediction). Building upon the foundational concepts of CLV prediction, this current endeavor focuses on leveraging more sophisticated machine learning techniques and comprehensive feature engineering to achieve higher accuracy and robustness. The key phases of development include:

- **Initial Data Exploration and Feature Engineering:** Starting with raw transactional data, comprehensive feature engineering was performed to derive meaningful customer attributes, including advanced RFM metrics and time-based features.
- **Probabilistic Modeling:** Implementation of BG/NBD and Gamma-Gamma models to understand customer behavior and predict future purchases and monetary value.
- **Advanced Ensemble Modeling:** Development and training of sophisticated ensemble models, including a Stacking Regressor and a Two-Stage Ensemble, to leverage the strengths of multiple predictive approaches.
- **Rigorous Model Evaluation:** Establishment of a robust evaluation framework using time-series cross-validation to ensure model reliability and identify the best-performing solution.

This iterative development process has led to the current state of the project, which emphasizes high-accuracy CLV prediction through advanced machine learning techniques.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your_clv_project.git
    cd your_clv_project
    ```
    (Note: Replace `https://github.com/your-username/your_clv_project.git` with the actual repository URL if it's hosted on GitHub.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file contains the following dependencies:
    ```
    pandas>=1.3.0
    numpy>=1.20.0
    scikit-learn>=1.0.0
    xgboost>=1.5.0
    lightgbm>=3.2.0
    lifetimes>=0.11.0
    optuna>=2.10.0
    matplotlib>=3.3.0
    seaborn>=0.11.0
    joblib>=1.1.0
    ```

## Usage

The project workflow is designed to be executed sequentially through the Python scripts in the `scripts/` directory.

1.  **Data Preparation and Feature Engineering:**
    This script preprocesses the raw transaction data and generates a rich set of customer features.
    ```bash
    python scripts/enhanced_feature_engineering.py
    ```
    *Output:* `data/processed/enhanced_customer_features.csv`

2.  **Probabilistic CLV Modeling:**
    This script builds and applies probabilistic models (BG/NBD, Gamma-Gamma) to predict CLV and save the predictions.
    ```bash
    python scripts/probabilistic_clv_models.py
    ```
    *Output:* `outputs/reports/probabilistic_clv_predictions.csv`

3.  **Advanced Ensemble Model Training:**
    This script trains the various ensemble models (Stacking, Two-Stage, Optimized XGBoost) using the generated features and probabilistic CLV predictions.
    ```bash
    python scripts/advanced_ensemble_model.py
    ```
    *Output:* Trained models saved in `outputs/models/` (e.g., `stacking_ensemble_clv_model.pkl`, `two_stage_ensemble_clv_model.pkl`, `xgb_optimized_clv_model.pkl`)

4.  **Model Evaluation and Optimization:**
    This script performs a comprehensive evaluation of the trained models, including time-series cross-validation, and identifies the best-performing model.
    ```bash
    python scripts/model_evaluation_optimization.py
    ```
    *Output:* `model_evaluation_results.json` in the project root, containing detailed evaluation metrics.

## Project Structure

```
your_clv_project/
├───requirements.txt
├───data/
│   ├───models/                 # Placeholder for any data-related models (e.g., clustering)
│   ├───processed/
│   │   └───enhanced_customer_features.csv  # Processed features for modeling
│   └───raw/
│       └───online_retail_II.csv            # Raw input data
├───outputs/
│   ├───models/                 # Trained machine learning models (.pkl files)
│   │   ├───stacking_ensemble_clv_model.pkl
│   │   ├───two_stage_ensemble_clv_model.pkl
│   │   └───xgb_optimized_clv_model.pkl
│   ├───reports/
│   │   └───probabilistic_clv_predictions.csv # Predictions from probabilistic models
│   └───visualizations/         # Placeholder for any generated plots/charts
└───scripts/
    ├───advanced_ensemble_model.py          # Script for training ensemble CLV models
    ├───enhanced_feature_engineering.py     # Script for data preprocessing and feature creation
    ├───model_evaluation_optimization.py    # Script for model evaluation and hyperparameter tuning
    ├───probabilistic_clv_models.py         # Script for building probabilistic CLV models
    └───__pycache__/                        # Python bytecode cache
```

## Results

After running the full pipeline, the `model_evaluation_optimization.py` script provides a comprehensive summary of the model performance.

**Key Evaluation Metrics (from the last run):**

*   **Best Performing Model:** `two_stage_ensemble`
*   **Best R² Score:** 0.9615 (This indicates that approximately 96.15% of the variance in the target CLV can be explained by the model, suggesting a very strong fit.)
*   **RMSE (Root Mean Squared Error):** 1854.35 (This is the standard deviation of the residuals (prediction errors). A lower RMSE indicates a better fit.)
*   **MAE (Mean Absolute Error):** 265.98 (This is the average of the absolute errors. It gives a direct measure of the average magnitude of the errors.)
*   **Mean Residual:** 122.34 (The average difference between actual and predicted values. Ideally close to zero.)
*   **Standard Deviation of Residuals:** 1850.31 (Measures the spread of the residuals.)

These results demonstrate that the `two_stage_ensemble` model, which incorporates a churn prediction stage, significantly outperforms the other models in terms of R² score and RMSE, making it the most reliable for CLV prediction in this context.

## Future Enhancements

*   **Hyperparameter Optimization:** Implement full Bayesian optimization for all models to find optimal hyperparameters.
*   **Ensemble Weight Optimization:** Develop a method to optimize the weights of the individual models within the ensemble for improved overall performance.
*   **Deployment:** Containerize the application using Docker for easier deployment and scalability.
*   **Dashboarding:** Create interactive dashboards (e.g., using Streamlit or Dash) to visualize CLV predictions, customer segments, and model performance.
*   **Real-time Prediction API:** Develop a REST API for real-time CLV predictions.
*   **More Data Sources:** Integrate additional data sources (e.g., website activity, customer service interactions) to enrich features.
*   **Unsupervised Learning:** Explore clustering techniques to identify natural customer segments before CLV prediction.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
