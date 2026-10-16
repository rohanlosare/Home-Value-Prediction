import os
import pickle
import warnings
from flask import Flask, request, render_template_string

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Keep linear.pkl in the same folder as this app.py file.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear.pkl")

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    model = None

# These mappings assume the categorical values were encoded alphabetically
# with sklearn LabelEncoder:
# sex: female=0, male=1 | smoker: no=0, yes=1
# region: northeast=0, northwest=1, southeast=2, southwest=3
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Medical Cost Predictor</title>
  <style>
    :root { --navy:#10213a; --blue:#2563eb; --cyan:#06b6d4; --ink:#172033; --muted:#64748b; }
    * { box-sizing:border-box; }
    body {
      margin:0; min-height:100vh; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink);
      background:radial-gradient(circle at 8% 10%,#dbeafe 0,transparent 32%),
                 radial-gradient(circle at 92% 85%,#cffafe 0,transparent 30%),#f5f8ff;
      padding:32px 16px;
    }
    .wrap { max-width:1050px; margin:auto; }
    .top { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:26px; }
    .brand { display:flex; align-items:center; gap:12px; }
    .logo { width:48px; height:48px; display:grid; place-items:center; color:#fff; font-size:25px;
            border-radius:15px; background:linear-gradient(135deg,var(--blue),var(--cyan)); box-shadow:0 8px 22px #2563eb35; }
    h1 { font-size:clamp(24px,4vw,34px); margin:0; letter-spacing:-.7px; color:var(--navy); }
    .subtitle { color:var(--muted); margin:6px 0 0; font-size:14px; }
    .badge { border:1px solid #bfdbfe; background:#eff6ff; color:#1d4ed8; border-radius:99px; padding:9px 13px; font-size:12px; font-weight:700; white-space:nowrap; }
    .grid { display:grid; grid-template-columns:1.05fr .95fr; gap:22px; align-items:stretch; }
    .card { background:rgba(255,255,255,.92); border:1px solid #e5eaf3; border-radius:24px; padding:28px; box-shadow:0 18px 50px #1e3a8a0d; }
    .card h2 { margin:0 0 7px; font-size:21px; color:var(--navy); }
    .help { margin:0 0 24px; font-size:13px; line-height:1.6; color:var(--muted); }
    .fields { display:grid; grid-template-columns:1fr 1fr; gap:17px; }
    label { display:block; font-size:13px; font-weight:700; color:#334155; margin-bottom:8px; }
    input, select { width:100%; border:1px solid #d8e0ec; border-radius:12px; padding:13px 12px; font-size:14px; color:#172033; background:#fff; outline:none; transition:.2s; }
    input:focus, select:focus { border-color:#60a5fa; box-shadow:0 0 0 4px #dbeafe; }
    .full { grid-column:1/-1; }
    button { width:100%; border:0; border-radius:13px; padding:15px 18px; margin-top:22px; color:#fff; font-size:15px; font-weight:800; cursor:pointer; background:linear-gradient(110deg,var(--blue),var(--cyan)); box-shadow:0 10px 22px #2563eb30; transition:transform .15s,filter .15s; }
    button:hover { transform:translateY(-1px); filter:saturate(1.15); }
    .result { margin-top:20px; padding:19px; border-radius:16px; background:#ecfdf5; border:1px solid #a7f3d0; }
    .result .eyebrow { font-size:11px; text-transform:uppercase; letter-spacing:1.3px; color:#047857; font-weight:800; }
    .amount { font-size:clamp(27px,4vw,36px); font-weight:850; color:#065f46; margin:7px 0 5px; letter-spacing:-1px; overflow-wrap:anywhere; }
    .result p { margin:0; color:#047857; font-size:12px; line-height:1.5; }
    .error { margin-top:18px; padding:13px 15px; border-radius:12px; background:#fff1f2; border:1px solid #fecdd3; color:#9f1239; font-size:13px; }
    .side { color:#fff; background:linear-gradient(150deg,#10213a 0%,#173b69 58%,#087e9b 130%); position:relative; overflow:hidden; }
    .side:after { content:""; position:absolute; width:240px; height:240px; border:1px solid #ffffff1b; border-radius:50%; right:-100px; top:-90px; box-shadow:0 0 0 28px #ffffff08,0 0 0 58px #ffffff06; }
    .side h2 { color:#fff; font-size:25px; line-height:1.25; max-width:300px; margin:16px 0 12px; }
    .side p { color:#dbeafe; line-height:1.7; font-size:14px; }
    .feature { display:flex; gap:12px; margin-top:24px; align-items:flex-start; }
    .check { flex:0 0 31px; height:31px; display:grid; place-items:center; border-radius:10px; background:#ffffff1c; color:#67e8f9; font-weight:900; }
    .feature strong { display:block; font-size:14px; margin:1px 0 5px; }
    .feature span { display:block; font-size:12px; line-height:1.5; color:#bfdbfe; }
    .disclaimer { margin-top:24px; padding-top:17px; border-top:1px solid #ffffff25; color:#bfdbfe; font-size:11px; line-height:1.6; }
    .foot { text-align:center; color:#8491a7; font-size:11px; margin:22px 0 0; }
    @media(max-width:780px) { .grid{grid-template-columns:1fr;} .side{order:2;} .top{align-items:flex-start;} .badge{display:none;} .card{padding:22px;} }
    @media(max-width:420px) { .fields{grid-template-columns:1fr;} .full{grid-column:auto;} }
  </style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <div class="logo">✚</div>
      <div><h1>Medical Cost Predictor</h1><p class="subtitle">Estimate medical insurance charges in seconds</p></div>
    </div>
    <div class="badge">✦ MACHINE LEARNING</div>
  </header>

  <main class="grid">
    <section class="card">
      <h2>Enter your details</h2>
      <p class="help">Fill in the information below to get an estimated insurance charge.</p>
      {% if not model_ready %}
        <div class="error">Model file not found. Put your trained model in this folder and name it <b>linear.pkl</b>.</div>
      {% endif %}
      <form method="POST" action="/">
        <div class="fields">
          <div>
            <label for="age">Age (years)</label>
            <input id="age" name="age" type="number" min="18" max="100" value="{{ values.get('age','30') }}" required>
          </div>
          <div>
            <label for="sex">Sex</label>
            <select id="sex" name="sex" required>
              <option value="female" {% if values.get('sex','female')=='female' %}selected{% endif %}>Female</option>
              <option value="male" {% if values.get('sex')=='male' %}selected{% endif %}>Male</option>
            </select>
          </div>
          <div>
            <label for="bmi">BMI</label>
            <input id="bmi" name="bmi" type="number" min="10" max="80" step="0.1" value="{{ values.get('bmi','25.0') }}" required>
          </div>
          <div>
            <label for="children">Number of children</label>
            <input id="children" name="children" type="number" min="0" max="20" value="{{ values.get('children','0') }}" required>
          </div>
          <div>
            <label for="smoker">Smoker?</label>
            <select id="smoker" name="smoker" required>
              <option value="no" {% if values.get('smoker','no')=='no' %}selected{% endif %}>No</option>
              <option value="yes" {% if values.get('smoker')=='yes' %}selected{% endif %}>Yes</option>
            </select>
          </div>
          <div>
            <label for="region">Region</label>
            <select id="region" name="region" required>
              {% for r in ['northeast','northwest','southeast','southwest'] %}
              <option value="{{ r }}" {% if values.get('region','northeast')==r %}selected{% endif %}>{{ r.title() }}</option>
              {% endfor %}
            </select>
          </div>
        </div>
        <button type="submit">Calculate Estimated Cost&nbsp; →</button>
      </form>
      {% if prediction is not none %}
      <div class="result" aria-live="polite">
        <div class="eyebrow">Estimated insurance charge</div>
        <div class="amount">{{ currency }}{{ "{:,.2f}".format(prediction) }}</div>
        <p>This is a model-based estimate, not a quote or medical/financial advice.</p>
      </div>
      {% endif %}
      {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </section>

    <aside class="card side">
      <div style="font-size:12px;font-weight:800;letter-spacing:1.5px;color:#67e8f9;">SMART • SIMPLE • FAST</div>
      <h2>Understand your estimated medical costs.</h2>
      <p>This demo uses a trained Linear Regression model to estimate insurance charges from six input features.</p>
      <div class="feature"><div class="check">✓</div><div><strong>Quick estimate</strong><span>Submit the form to calculate a prediction instantly.</span></div></div>
      <div class="feature"><div class="check">✓</div><div><strong>Simple inputs</strong><span>Age, BMI, family details, smoking status, and region.</span></div></div>
      <div class="feature"><div class="check">✓</div><div><strong>Machine learning</strong><span>Predictions are generated by your saved model file.</span></div></div>
      <div class="disclaimer">Important: results depend on how the model was trained and how categorical values were encoded. For educational demonstration only; actual charges may differ.</div>
    </aside>
  </main>
  <div class="foot">Medical Cost Predictor · Flask + Scikit-learn</div>
</div>
</body>
</html>
"""

# LabelEncoder-style alphabetical mappings. Confirm these match your training code.
SEX_MAP = {"female": 0, "male": 1}
SMOKER_MAP = {"no": 0, "yes": 1}
REGION_MAP = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}


@app.route("/", methods=["GET", "POST"])
def home():
    values = {
        "age": "30", "sex": "female", "bmi": "25.0",
        "children": "0", "smoker": "no", "region": "northeast"
    }
    prediction = None
    error = None

    if request.method == "POST":
        values = {key: request.form.get(key, "").strip()
                  for key in ["age", "sex", "bmi", "children", "smoker", "region"]}
        try:
            if model is None:
                raise FileNotFoundError("linear.pkl was not found beside app.py.")
            age = int(values["age"])
            bmi = float(values["bmi"])
            children = int(values["children"])
            sex = values["sex"].lower()
            smoker = values["smoker"].lower()
            region = values["region"].lower()

            if not 18 <= age <= 100:
                raise ValueError("Age must be between 18 and 100.")
            if not 10 <= bmi <= 80:
                raise ValueError("BMI must be between 10 and 80.")
            if not 0 <= children <= 20:
                raise ValueError("Number of children must be between 0 and 20.")
            if sex not in SEX_MAP or smoker not in SMOKER_MAP or region not in REGION_MAP:
                raise ValueError("Please choose valid options for all categorical fields.")

            # Feature order matches the model's feature_names_in_:
            # age, sex, bmi, children, smoker, region
            features = [[age, SEX_MAP[sex], bmi, children, SMOKER_MAP[smoker], REGION_MAP[region]]]
            prediction = float(model.predict(features)[0])
            if prediction < 0:
                prediction = 0.0
        except (ValueError, TypeError, FileNotFoundError) as exc:
            error = str(exc) if str(exc) else "Please check the entered values and try again."
        except Exception:
            error = "The model could not make a prediction. Check that the model and feature encoding match."

    return render_template_string(
        HTML, values=values, prediction=prediction, error=error,
        model_ready=(model is not None), currency="₹"
    )


if __name__ == "__main__":
    # Render supplies PORT automatically. Debug stays off in production.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
