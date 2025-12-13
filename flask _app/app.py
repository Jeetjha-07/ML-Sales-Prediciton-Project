from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# Global variables to store models and scalers
models = {}
sample_data = {}
original_data = None

def load_and_train_models():
    """Load data and train all three models"""
    global models, sample_data, original_data
    
    # Load the dataset
    data = pd.read_excel('Adidas US Sales Datasets.xlsx')
    original_data = data.copy()
    
    # Store sample data for frontend (keep original categorical values)
    sample_data['raw'] = data.head(10).to_dict('records')
    
    # ==================== MODEL 1: Linear Regression for Total Sales ====================
    print("Training Model 1: Linear Regression...")
    data1 = data.copy()
    data1 = data1.drop(columns=["Retailer ID", "Operating Margin", "Invoice Date"], errors='ignore')
    
    # Store categorical mappings for Model 1
    cat_mappings_1 = {}
    cat_cols = data1.select_dtypes(include=['object']).columns
    for col in cat_cols:
        frequency = data1[col].value_counts() / len(data1)
        cat_mappings_1[col] = frequency.to_dict()
        data1[col] = data1[col].map(frequency)
    
    # Remove outliers
    q1 = data1["Total Sales"].quantile(0.01)
    q99 = data1["Total Sales"].quantile(0.99)
    data1["Total Sales"] = data1["Total Sales"].clip(q1, q99)
    
    # Log transform
    data1['Log_Total_Sales'] = np.log1p(data1['Total Sales'])
    data1['Log_Operating_Profit'] = np.log1p(data1['Operating Profit'])
    data1.drop(columns=['Total Sales', 'Operating Profit'], inplace=True)
    
    X1 = data1.drop("Log_Total_Sales", axis=1)
    y1 = data1["Log_Total_Sales"]
    
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    
    scaler1 = StandardScaler()
    X1_train_scaled = scaler1.fit_transform(X1_train)
    X1_test_scaled = scaler1.transform(X1_test)
    
    lr = LinearRegression()
    lr.fit(X1_train_scaled, y1_train)
    
    y1_pred = lr.predict(X1_test_scaled)
    
    models['model1'] = {
        'model': lr,
        'scaler': scaler1,
        'cat_mappings': cat_mappings_1,
        'feature_names': X1.columns.tolist(),
        'metrics': {
            'mae': float(mean_absolute_error(y1_test, y1_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y1_test, y1_pred))),
            'r2': float(r2_score(y1_test, y1_pred))
        },
        'test_data': {'y_test': y1_test.values[:50].tolist(), 'y_pred': y1_pred[:50].tolist()}
    }
    
    # ==================== MODEL 2: Decision Tree for Sales Method ====================
    print("Training Model 2: Decision Tree Classifier...")
    data2 = data.copy()
    
    sales_method_map = {"In-store": 0, "Outlet": 1, "Online": 2}
    data2['Sales_Method_Encoded'] = data2['Sales Method'].map(sales_method_map)
    
    X2 = data2.drop(columns=['Sales Method', 'Sales_Method_Encoded', 'Invoice Date',
                             'Retailer ID', 'Operating Margin'], errors='ignore')
    y2 = data2['Sales_Method_Encoded']
    
    cat_cols2 = X2.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in cat_cols2:
        le = LabelEncoder()
        X2[col] = le.fit_transform(X2[col])
        le_dict[col] = {val: idx for idx, val in enumerate(le.classes_)}
    
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
    
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=None, splitter='best', random_state=42)
    dt.fit(X2_train, y2_train)
    
    y2_pred = dt.predict(X2_test)
    
    models['model2'] = {
        'model': dt,
        'label_encoders': le_dict,
        'feature_names': X2.columns.tolist(),
        'class_names': ["In-store", "Outlet", "Online"],
        'metrics': {
            'accuracy': float(accuracy_score(y2_test, y2_pred)),
            'confusion_matrix': confusion_matrix(y2_test, y2_pred).tolist()
        },
        'test_data': {'y_test': y2_test.values.tolist(), 'y_pred': y2_pred.tolist()}
    }
    
    # ==================== MODEL 3: Random Forest for Units Sold ====================
    print("Training Model 3: Random Forest Regressor...")
    data3 = data.copy()
    
    cat_cols3 = data3.select_dtypes(include=['object']).columns
    le3_dict = {}
    for col in cat_cols3:
        le3 = LabelEncoder()
        data3[col] = le3.fit_transform(data3[col])
        le3_dict[col] = {val: idx for idx, val in enumerate(le3.classes_)}
    
    data3['Year'] = data3["Invoice Date"].dt.year
    data3['Month'] = data3["Invoice Date"].dt.month
    data3['Weekday'] = data3["Invoice Date"].dt.weekday
    
    # Note: Including Total Sales creates data leakage but matches your original model
    X3 = data3.drop(columns=["Units Sold", "Invoice Date", "Retailer ID"], errors='ignore')
    y3 = data3['Units Sold']
    
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X3_train, y3_train)
    
    y3_pred = rf.predict(X3_test)
    
    models['model3'] = {
        'model': rf,
        'label_encoders': le3_dict,
        'feature_names': X3.columns.tolist(),
        'metrics': {
            'mae': float(mean_absolute_error(y3_test, y3_pred)),
            'mse': float(mean_squared_error(y3_test, y3_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y3_test, y3_pred))),
            'r2': float(r2_score(y3_test, y3_pred))
        },
        'test_data': {'y_test': y3_test.values[:50].tolist(), 'y_pred': y3_pred[:50].tolist()},
        'feature_importance': {str(k): float(v) for k, v in zip(X3.columns, rf.feature_importances_)}
    }
    
    print("All models trained successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/model-info/<model_id>')
def get_model_info(model_id):
    """Get model information and metrics"""
    if model_id not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    model_data = models[model_id]
    return jsonify({
        'feature_names': model_data['feature_names'],
        'metrics': model_data['metrics'],
        'class_names': model_data.get('class_names', None)
    })

@app.route('/api/predict/<model_id>', methods=['POST'])
def predict(model_id):
    """Make prediction using specified model"""
    if model_id not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    data = request.json
    model_data = models[model_id]
    
    try:
        if model_id == 'model1':
            # Linear Regression prediction - handle categorical encoding
            features = []
            for fname in model_data['feature_names']:
                val = data.get(fname, 0)
                
                # Apply frequency encoding for categorical columns
                if fname in model_data['cat_mappings']:
                    # Get the frequency value for this category
                    val = model_data['cat_mappings'][fname].get(val, 0.0)
                
                features.append(float(val))
            
            features_scaled = model_data['scaler'].transform([features])
            log_prediction = model_data['model'].predict(features_scaled)[0]
            prediction = np.expm1(log_prediction)
            
            return jsonify({
                'prediction': f"${prediction:,.2f}",
                'raw_value': float(prediction),
                'confidence': float(model_data['metrics']['r2'] * 100)
            })
        
        elif model_id == 'model2':
            # Decision Tree classification
            features = []
            for fname in model_data['feature_names']:
                val = data.get(fname, 0)
                if fname in model_data['label_encoders']:
                    val = model_data['label_encoders'][fname].get(val, 0)
                features.append(float(val))
            
            pred_idx = int(model_data['model'].predict([features])[0])
            prediction = model_data['class_names'][pred_idx]
            probabilities = model_data['model'].predict_proba([features])[0]
            
            return jsonify({
                'prediction': prediction,
                'confidence': float(max(probabilities) * 100),
                'probabilities': {name: float(prob * 100) for name, prob in zip(model_data['class_names'], probabilities)}
            })
        
        elif model_id == 'model3':
            # Random Forest regression
            features = []
            for fname in model_data['feature_names']:
                val = data.get(fname, 0)
                if fname in model_data['label_encoders']:
                    val = model_data['label_encoders'][fname].get(val, 0)
                features.append(float(val))
            
            prediction = model_data['model'].predict([features])[0]
            
            return jsonify({
                'prediction': f"{int(prediction):,} units",
                'raw_value': float(prediction),
                'confidence': float(model_data['metrics']['r2'] * 100)
            })
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualization/<model_id>')
def get_visualization(model_id):
    """Generate visualization for specified model"""
    if model_id not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    model_data = models[model_id]
    
    try:
        plt.style.use('dark_background')
        
        if model_id == 'model1':
            # Actual vs Predicted plot
            fig, ax = plt.subplots(figsize=(12, 6))
            y_test = model_data['test_data']['y_test']
            y_pred = model_data['test_data']['y_pred']
            
            ax.plot(y_test, label='Actual', marker='o', linewidth=2.5, markersize=8, color='#60A5FA')
            ax.plot(y_pred, label='Predicted', marker='x', linewidth=2.5, markersize=8, color='#F472B6')
            ax.set_title('Actual vs Predicted Total Sales', fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Sample Index', fontsize=14)
            ax.set_ylabel('Log(Total Sales)', fontsize=14)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.2, linestyle='--')
            
        elif model_id == 'model2':
            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = np.array(model_data['metrics']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=model_data['class_names'],
                       yticklabels=model_data['class_names'], ax=ax,
                       cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
            ax.set_title('Confusion Matrix - Sales Method Prediction', fontsize=18, fontweight='bold', pad=20)
            ax.set_ylabel('Actual', fontsize=14)
            ax.set_xlabel('Predicted', fontsize=14)
            
        elif model_id == 'model3':
            # Feature Importance
            fig, ax = plt.subplots(figsize=(12, 8))
            feature_imp = model_data['feature_importance']
            sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importance = zip(*sorted_features)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
            bars = ax.barh(range(len(features)), importance, color=colors)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=12)
            ax.set_xlabel('Importance Score', fontsize=14)
            ax.set_title('Top 10 Feature Importance - Units Sold Prediction', fontsize=18, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.2, axis='x', linestyle='--')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{importance[i]:.3f}', 
                       ha='left', va='center', fontsize=10, color='white', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='#1e1e2e')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return jsonify({'image': f'data:image/png;base64,{image_base64}'})
    
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/sample-data')
def get_sample_data():
    """Get sample data for testing"""
    return jsonify(sample_data)

if __name__ == '__main__':
    print("=" * 60)
    print("Loading dataset and training models...")
    print("=" * 60)
    load_and_train_models()
    print("\n" + "=" * 60)
    print("🚀 Flask server starting...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)