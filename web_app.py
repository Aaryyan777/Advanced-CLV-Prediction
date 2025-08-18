from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import threading
import subprocess
import sys
import json
import numpy as np
from scripts.advanced_ensemble_model import TwoStageEnsemble, AdvancedStackingRegressor

app = Flask(__name__)

# Global variable to track pipeline status
pipeline_status = {'running': False, 'message': 'Idle'}

def run_script(script_name, step_message):
    global pipeline_status
    pipeline_status['message'] = step_message
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', script_name)
    
    # Use sys.executable to ensure the correct python interpreter is used
    process = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    
    if process.returncode != 0:
        raise Exception(f"Script {script_name} failed: {process.stderr}")
    return process.stdout

def run_clv_pipeline_task():
    global pipeline_status
    pipeline_status = {'running': True, 'message': 'Starting pipeline...'}
    try:
        # Ensure output directories exist
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('outputs/models', exist_ok=True)
        os.makedirs('outputs/reports', exist_ok=True)

        # 1. Enhanced Feature Engineering
        run_script('enhanced_feature_engineering.py', 'Step 1: Enhanced Feature Engineering...')

        # 2. Probabilistic CLV Modeling
        run_script('probabilistic_clv_models.py', 'Step 2: Probabilistic CLV Modeling...')

        # 3. Advanced Ensemble Model Training
        run_script('advanced_ensemble_model.py', 'Step 3: Advanced Ensemble Model Training...')

        # 4. Model Evaluation and Optimization
        run_script('model_evaluation_optimization.py', 'Step 4: Model Evaluation and Optimization...')

        # 5. Customer Segmentation
        run_script('k-means_clustering.py', 'Step 5: Customer Segmentation...')

        pipeline_status['message'] = 'Pipeline finished successfully!'
    except Exception as e:
        pipeline_status['message'] = f'Pipeline failed: {str(e)}'
    finally:
        pipeline_status['running'] = False

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_clv_value = None
    model_results = None
    cluster_results = None

    if request.method == 'POST':
        if 'action' in request.form and request.form['action'] == 'predict':
            try:
                recency = float(request.form['recency'])
                frequency = float(request.form['frequency'])
                monetary = float(request.form['monetary'])

                # Load the best model (two_stage_ensemble_clv_model.pkl)
                model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'models', 'two_stage_ensemble_clv_model.pkl')
                model = joblib.load(model_path)

                # Load enhanced features to get a sample customer structure
                features_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'enhanced_customer_features.csv')
                features_df = pd.read_csv(features_path)
                
                # Prepare features for prediction, mirroring advanced_ensemble_model.py
                # Exclude non-numeric columns and the target variable itself
                numeric_features_df = features_df.select_dtypes(include=np.number)
                feature_columns = [col for col in numeric_features_df.columns 
                              if col not in ['CustomerID']]
                
                # Create a sample from the training data and update with user's input
                sample_customer = numeric_features_df[feature_columns].iloc[0].to_dict()
                sample_customer['Recency'] = recency
                sample_customer['Frequency'] = frequency
                sample_customer['Monetary'] = monetary # This will be used by the model, even if not in feature_columns for prediction

                customer_df = pd.DataFrame([sample_customer])
                
                # Ensure the order of columns matches the training data
                customer_df = customer_df[feature_columns]

                predicted_clv_value = model.predict(customer_df)[0]

            except FileNotFoundError:
                predicted_clv_value = "Error: Model or processed data not found. Please run the pipeline first."
            except Exception as e:
                predicted_clv_value = f"Error: {str(e)}"

    # Load results for display on the single page
    # Model Evaluation Results
    try:
        results_path = os.path.join(os.path.dirname(__file__), 'model_evaluation_results.json')
        with open(results_path, 'r') as f:
            results_data = json.load(f)
            model_comparison_df = pd.DataFrame(results_data['model_comparison']).T
            model_results = model_comparison_df.to_html(classes='table table-striped')
            
            best_metrics_df = pd.DataFrame([results_data['best_metrics']])
            model_results += "<h3 style=\"margin-top: 20px;\">Best Model Metrics:</h3>"
            model_results += best_metrics_df.to_html(classes='table table-striped')

    except FileNotFoundError:
        model_results = "Model evaluation results not found. Please run the pipeline first."
    except Exception as e:
        model_results = f"Error loading model results: {str(e)}"

    # Customer Segmentation Analysis
    try:
        cluster_path = os.path.join(os.path.dirname(__file__), 'outputs', 'reports', 'cluster_analysis.csv')
        cluster_results = pd.read_csv(cluster_path).to_html(classes='table table-striped')
    except FileNotFoundError:
        cluster_results = "Customer segmentation analysis not found. Please run the pipeline first."
    except Exception as e:
        cluster_results = f"Error loading cluster results: {str(e)}"

    return render_template('index.html', 
                           predicted_clv=predicted_clv_value,
                           model_results=model_results,
                           cluster_results=cluster_results)

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline_route():
    global pipeline_status
    if not pipeline_status['running']:
        thread = threading.Thread(target=run_clv_pipeline_task)
        thread.start()
        return jsonify({'status': 'started', 'message': 'Pipeline started in background.'})
    return jsonify({'status': 'already_running', 'message': 'Pipeline is already running.'})

@app.route('/pipeline_status')
def get_pipeline_status():
    return jsonify(pipeline_status)

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    app.run(debug=True)
