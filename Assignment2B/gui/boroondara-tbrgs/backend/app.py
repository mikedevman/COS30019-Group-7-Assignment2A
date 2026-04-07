import os
import sys
import time
import warnings
import logging

# Suppress ALL ML backend logging (TensorFlow, oneDNN, absl)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_VERBOSITY'] = '-1'

# Disable loggers and filter warnings specifically
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*TensorFlow GPU support is not available.*')

from flask import Flask, request, jsonify
from flask_cors import CORS
from constants import DEFAULTS
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up from Assignment2B/gui/boroondara-tbrgs/backend to Assignment2B/
assignment2b_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

sys.path.append(assignment2b_dir)
sys.path.append(os.path.join(assignment2b_dir, 'integration'))

# Import the logic built previously
from integration.main import run_tbrgs

app = Flask(__name__)
# Enable CORS so the React frontend running on a different port can communicate with this API
CORS(app)

@app.route('/api/route', methods=['POST'])
def calculate_route():
    data = request.get_json()
    
    if not data or 'origin' not in data or 'destination' not in data:
        return jsonify({"error": "Please provide 'origin' and 'destination' SCATS nodes."}), 400
        
    origin_input = str(data['origin'])
    destination_input = str(data['destination'])
    model_name = str(data.get('model', 'lstm')).lower()
    is_bidirectional = bool(data.get('bidirectional', False))

    # Normalise to explicit model names
    if model_name == 'lstm' and is_bidirectional:
        model_name = 'bidirectional_lstm'

    elif model_name == 'gru' and is_bidirectional:
        model_name = 'bidirectional_gru'
    
    elif model_name == 'custom_gcn_lstm':
        model_name = 'custom_gcn_lstm'

    supported_models = ['lstm', 'bidirectional_lstm', 'gru', 'bidirectional_gru', 'custom_gcn_lstm']
    if model_name not in supported_models:
        model_name = 'lstm'
    
    # Extract new front-end parameters with defaults
    top_k = int(data.get('topK', DEFAULTS["topK"]))
    speed_limit = float(data.get('speedLimit', DEFAULTS["speedLimit"]))
    intersection_delay = float(data.get('intersectionDelay', DEFAULTS["intersectionDelay"]))
    
    # Print request details to console
    print(f"\n[POST /api/route] New Request")
    print(f"  From SCATS: {origin_input} -> To SCATS: {destination_input}")
    print(f"  Model: {model_name} | topK: {top_k} | Speed Limit: {speed_limit}km/h | Delay: {intersection_delay}s")

    origin = origin_input
    destination = destination_input
    map_path = os.path.join(assignment2b_dir, 'map_data', 'boroodara_tbrgs_map_coordinates.txt')
    
    try:
        # Switch CWD to ensure relative paths inside intergration.py resolve successfully
        os.chdir(assignment2b_dir)
        
        # run_tbrgs dynamically loads the ML model and calculates the top Yen's K Shortest Paths
        start_time = time.perf_counter()
        routes = run_tbrgs(
            filename=map_path, 
            origin=origin, 
            destination=destination, 
            model_name=model_name,
            k_routes=top_k,
            speed_limit=speed_limit,
            intersection_delay=intersection_delay
        )
        elapsed_time = time.perf_counter() - start_time
        
        # Print results to console
        print(f"  Found {len(routes)} routes in {elapsed_time:.3f}s:")
        for r in routes:
            nodes = len(r['path'])
            print(f"    - Route {r['route']} | {r['estimated_time_mins']} min | {r['distance_km']} km | {nodes} nodes")
        print("-" * 30)

        return jsonify({
            "status": "success",
            "origin": origin,
            "destination": destination,
            "model": model_name,
            "findingTimeSeconds": round(elapsed_time, 3),
            "routes": routes
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting TBRGS Machine Learning Backend API...")
    app.run(debug=True, port=5001)
