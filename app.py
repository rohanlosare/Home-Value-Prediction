import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained linear regression model
model = pickle.load(open('linear.pkl', 'rb'))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home-Value-Prediction</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 35px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 480px;
        }
        h2 {
            margin-top: 0;
            font-weight: 600;
            color: var(--text);
            text-align: center;
            margin-bottom: 24px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 6px;
            color: var(--text);
        }
        input, select {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }
        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result {
            margin-top: 24px;
            padding: 16px;
            background-color: #dbeafe;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            color: #1e40af;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Home-Value-Prediction</h2>
        <form method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" required min="1" max="100" placeholder="e.g. 25">
            </div>
            <div class="form-group">
                <label for="sex">Sex</label>
                <select id="sex" name="sex">
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label for="bmi">BMI</label>
                <input type="number" step="0.1" id="bmi" name="bmi" required placeholder="e.g. 24.5">
            </div>
            <div class="form-group">
                <label for="children">Children</label>
                <input type="number" id="children" name="children" required min="0" max="10" placeholder="e.g. 0">
            </div>
            <div class="form-group">
                <label for="smoker">Smoker</label>
                <select id="smoker" name="smoker">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label for="region">Region</label>
                <select id="region" name="region">
                    <option value="0">Northeast</option>
                    <option value="1">Northwest</option>
                    <option value="2">Southeast</option>
                    <option value="3">Southwest</option>
                </select>
            </div>
            <button type="submit">Predict Value</button>
        </form>
        {% if prediction is not none %}
        <div class="result">
            Predicted Value: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            bmi = float(request.form['bmi'])
            children = float(request.form['children'])
            smoker = float(request.form['smoker'])
            region = float(request.form['region'])
            
            features = np.array([[age, sex, bmi, children, smoker, region]])
            pred = model.predict(features)[0]
            prediction = f"₹ {round(pred, 2)}"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
