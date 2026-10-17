from flask import Flask, render_template_string, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Linear Regression model directly from the pickle file
with open('Linear.pkl', 'rb') as f:
    model = pickle.load(f)

# Modern UI template embedded directly inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Real Estate Valuation Suite</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex flex-col justify-between font-sans antialiased">

    <!-- Header Section -->
    <header class="w-full py-6 px-8 border-b border-slate-800/60 backdrop-blur-md bg-slate-900/40 sticky top-0 z-50">
        <div class="max-w-5xl mx-auto flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="h-10 w-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/30">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                    </svg>
                </div>
                <span class="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-indigo-200 to-indigo-400 bg-clip-text text-transparent">EstateValuate AI</span>
            </div>
            <span class="text-xs uppercase tracking-widest px-3 py-1 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 rounded-full font-medium">Linear Regression Engine</span>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-4xl mx-auto px-4 py-12 w-full flex-grow">
        <div class="text-center mb-10">
            <h1 class="text-4xl font-extrabold tracking-tight mb-3 text-white">Property Value Predictor</h1>
            <p class="text-slate-400 max-w-lg mx-auto text-sm">Enter the property specifications below to instantly calculate market valuation powered by machine learning.</p>
        </div>

        <div class="bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 shadow-2xl shadow-indigo-950/50">
            <form action="/predict" method="POST" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <!-- Square Footage -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Square Footage (sq ft)</label>
                        <input type="number" step="any" name="Square_Footage" required placeholder="e.g., 2100" 
                            value="{{ form_data.Square_Footage if form_data else '' }}"
                            class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Number of Bedrooms -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Bedrooms</label>
                        <input type="number" name="Num_Bedrooms" required placeholder="e.g., 3" 
                            value="{{ form_data.Num_Bedrooms if form_data else '' }}"
                            class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Number of Bathrooms -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Bathrooms</label>
                        <input type="number" step="any" name="Num_Bathrooms" required placeholder="e.g., 2.5" 
                            value="{{ form_data.Num_Bathrooms if form_data else '' }}"
                            class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Year Built -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Year Built</label>
                        <input type="number" name="Year_Built" required placeholder="e.g., 2012" 
                            value="{{ form_data.Year_Built if form_data else '' }}"
                            class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Lot Size -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Lot Size (sq ft)</label>
                        <input type="number" step="any" name="Lot_Size" required placeholder="e.g., 5000" 
                            value="{{ form_data.Lot_Size if form_data else '' }}"
                            class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>

                    <!-- Garage Size -->
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Garage Capacity (Cars)</label>
                        <input type="number" step="any" name="Garage_Size" required placeholder="e.g., 2" 
                            value="{{ form_data.Garage_Size if form_data else '' }}"
                            class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>

                </div>

                <!-- Neighborhood Quality -->
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-2">Neighborhood Quality Score (1 - 10)</label>
                    <input type="number" step="any" name="Neighborhood_Quality" required placeholder="e.g., 8" 
                        value="{{ form_data.Neighborhood_Quality if form_data else '' }}"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <!-- Submit Button -->
                <button type="submit" class="w-full py-4 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-bold rounded-xl shadow-lg shadow-indigo-600/30 transition duration-200 transform hover:-translate-y-0.5">
                    Calculate Estimated Valuation
                </button>
            </form>

            <!-- Results Banner -->
            {% if prediction_text %}
            <div class="mt-8 p-6 bg-slate-950/90 border border-indigo-500/30 rounded-2xl text-center shadow-inner">
                <p class="text-xs uppercase tracking-widest text-indigo-400 font-semibold mb-1">Estimated Market Value</p>
                <div class="text-3xl sm:text-4xl font-black text-white tracking-tight">{{ prediction_text }}</div>
            </div>
            {% endif %}
        </div>
    </main>

    <!-- Footer -->
    <footer class="w-full py-6 text-center text-xs text-slate-500 border-t border-slate-800/60 bg-slate-900/20">
        <p>&copy; 2026 EstateValuate AI. Built with Flask & Tailwind CSS.</p>
    </footer>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features aligned with the model parameters[cite: 1]
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        # Format input into array and predict
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
        
        formatted_output = f"${prediction:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=formatted_output, form_data=request.form)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}", form_data=request.form)

if __name__ == "__main__":
    app.run(debug=True)
