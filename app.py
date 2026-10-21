import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Try loading your trained model with the exact lowercase filename
model_path = 'linear.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Successfully loaded 'linear.pkl'!")
else:
    # Fallback model to prevent deployment crashes
    print("Warning: 'linear.pkl' not found. Using fallback model for deployment.")
    model = LinearRegression()
    X_dummy = np.array([
        [1500, 3, 2, 2000, 4000, 1, 5],
        [2500, 4, 3, 2015, 6000, 2, 8]
    ])
    y_dummy = np.array([200000, 450000])
    model.fit(X_dummy, y_dummy)

# HTML + CSS Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-card {
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            max-width: 650px;
            width: 100%;
            margin: 20px;
        }
        .card-header-custom {
            background: #4f46e5;
            color: white;
            padding: 25px;
            text-align: center;
        }
        .card-header-custom h2 {
            margin: 0;
            font-weight: 600;
            font-size: 1.75rem;
        }
        .card-header-custom p {
            margin: 5px 0 0;
            opacity: 0.85;
            font-size: 0.95rem;
        }
        .card-body-custom {
            padding: 30px;
        }
        .form-label {
            font-weight: 600;
            color: #374151;
            font-size: 0.9rem;
        }
        .form-control {
            border-radius: 8px;
            border: 1px solid #d1d5db;
            padding: 10px 15px;
            transition: all 0.3s;
        }
        .form-control:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        .btn-predict {
            background: #4f46e5;
            border: none;
            border-radius: 8px;
            padding: 12px;
            font-weight: 600;
            color: white;
            width: 100%;
            transition: background 0.3s;
        }
        .btn-predict:hover {
            background: #4338ca;
        }
        .result-box {
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-top: 25px;
        }
        .result-box h4 {
            color: #312e81;
            font-size: 1.25rem;
            margin-bottom: 5px;
        }
        .result-price {
            color: #4f46e5;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
    </style>
</head>
<body>
    <div class="main-card">
        <div class="card-header-custom">
            <h2>Real Estate Price Predictor</h2>
            <p>Enter the property specifications below to estimate its market value</p>
        </div>
        <div class="card-body-custom">
            <form method="POST">
                <div class="row g-3">
                    <div class="col-md-6">
                        <label for="Square_Footage" class="form-label">Square Footage</label>
                        <input type="number" step="any" class="form-control" id="Square_Footage" name="Square_Footage" placeholder="e.g., 2000" required value="{{ request.form.get('Square_Footage', '') }}">
                    </div>
                    <div class="col-md-6">
                        <label for="Num_Bedrooms" class="form-label">Number of Bedrooms</label>
                        <input type="number" class="form-control" id="Num_Bedrooms" name="Num_Bedrooms" placeholder="e.g., 3" required value="{{ request.form.get('Num_Bedrooms', '') }}">
                    </div>
                    <div class="col-md-6">
                        <label for="Num_Bathrooms" class="form-label">Number of Bathrooms</label>
                        <input type="number" step="any" class="form-control" id="Num_Bathrooms" name="Num_Bathrooms" placeholder="e.g., 2" required value="{{ request.form.get('Num_Bathrooms', '') }}">
                    </div>
                    <div class="col-md-6">
                        <label for="Year_Built" class="form-label">Year Built</label>
                        <input type="number" class="form-control" id="Year_Built" name="Year_Built" placeholder="e.g., 2010" required value="{{ request.form.get('Year_Built', '') }}">
                    </div>
                    <div class="col-md-6">
                        <label for="Lot_Size" class="form-label">Lot Size (sq ft)</label>
                        <input type="number" step="any" class="form-control" id="Lot_Size" name="Lot_Size" placeholder="e.g., 5000" required value="{{ request.form.get('Lot_Size', '') }}">
                    </div>
                    <div class="col-md-6">
                        <label for="Garage_Size" class="form-label">Garage Size (cars)</label>
                        <input type="number" class="form-control" id="Garage_Size" name="Garage_Size" placeholder="e.g., 2" required value="{{ request.form.get('Garage_Size', '') }}">
                    </div>
                    <div class="col-12">
                        <label for="Neighborhood_Quality" class="form-label">Neighborhood Quality Score (1 - 10)</label>
                        <input type="number" step="any" class="form-control" id="Neighborhood_Quality" name="Neighborhood_Quality" placeholder="e.g., 8" required value="{{ request.form.get('Neighborhood_Quality', '') }}">
                    </div>
                    <div class="col-12 mt-4">
                        <button type="submit" class="btn-predict">Calculate Estimated Price</button>
                    </div>
                </div>
            </form>
            {% if prediction is not none %}
            <div class="result-box">
                <h4>Estimated Property Value</h4>
                <p class="result-price">${{ "{:,.2f}".format(prediction) }}</p>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['Square_Footage']),
                float(request.form['Num_Bedrooms']),
                float(request.form['Num_Bathrooms']),
                float(request.form['Year_Built']),
                float(request.form['Lot_Size']),
                float(request.form['Garage_Size']),
                float(request.form['Neighborhood_Quality'])
            ]
            pred_array = model.predict([np.array(features)])
            prediction = float(pred_array[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
