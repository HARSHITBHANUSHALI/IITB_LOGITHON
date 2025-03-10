from flask import Flask, request, jsonify
import requests
import json
from typing import Dict, Optional

app = Flask(__name__)

# URL to a public data source (example using a GitHub-hosted JSON file)
# You can replace this with any public API or data source
CARBON_FACTORS_URL = "https://raw.githubusercontent.com/example/repo/main/carbon-factors.json"

# Cache for carbon factors
carbon_factors: Optional[Dict] = None

def fetch_carbon_factors() -> Dict:
    """Fetch carbon factors from a public source"""
    global carbon_factors
    try:
        if carbon_factors is None:
            response = requests.get(CARBON_FACTORS_URL, timeout=5)
            response.raise_for_status()
            carbon_factors = response.json()
        return carbon_factors
    except requests.RequestException as e:
        # Fallback factors in case the source is unavailable
        fallback = {
            'car_miles': 0.403,        # kg CO2e per mile
            'electricity_kwh': 0.475,  # kg CO2e per kWh
            'meat_meals': 3.3,         # kg CO2e per meat meal
            'vegan_meals': 1.2         # kg CO2e per vegan meal
        }
        print(f"Failed to fetch carbon factors: {str(e)}. Using fallback values.")
        carbon_factors = fallback
        return fallback

@app.route('/api/carbon-footprint', methods=['POST'])
def calculate_carbon_footprint():
    try:
        # Fetch latest carbon factors
        factors = fetch_carbon_factors()
        
        # Get data from request
        data = request.get_json()
        
        # Extract values (with defaults if not provided)
        car_miles = float(data.get('car_miles', 0))
        electricity_kwh = float(data.get('electricity_kwh', 0))
        meat_meals = float(data.get('meat_meals', 0))
        vegan_meals = float(data.get('vegan_meals', 0))
        
        # Calculate carbon footprint (converting to annual values)
        car_emissions = car_miles * factors['car_miles'] * 52
        electricity_emissions = electricity_kwh * factors['electricity_kwh'] * 12
        meat_emissions = meat_meals * factors['meat_meals'] * 52
        vegan_emissions = vegan_meals * factors['vegan_meals'] * 52
        
        # Total annual carbon footprint in kg CO2e
        total_footprint = car_emissions + electricity_emissions + meat_emissions + vegan_emissions
        
        # Prepare response
        result = {
            'annual_carbon_footprint_kg': round(total_footprint, 2),
            'breakdown': {
                'transportation': round(car_emissions, 2),
                'electricity': round(electricity_emissions, 2),
                'diet': round(meat_emissions + vegan_emissions, 2)
            },
            'unit': 'kg CO2e per year',
            'source': 'Public data (fallback used if fetch failed)'
        }
        
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
        
    except (ValueError, TypeError) as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid input: please provide numeric values'
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    factors = fetch_carbon_factors()
    return jsonify({
        'message': 'Welcome to Carbon Footprint Calculator API',
        'endpoint': '/api/carbon-footprint',
        'method': 'POST',
        'expected_input': {
            'car_miles': 'miles driven per week (float)',
            'electricity_kwh': 'electricity usage in kWh per month (float)',
            'meat_meals': 'number of meat-based meals per week (float)',
            'vegan_meals': 'number of vegan meals per week (float)'
        },
        'current_factors': factors
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1000)