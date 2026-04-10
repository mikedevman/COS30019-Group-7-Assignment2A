import os
import sys
import time
import warnings
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from constants import DEFAULTS
import traceback

# Suppress noisy TensorFlow and related ML backend logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_VERBOSITY'] = '-1'

# Disable TensorFlow logger and filter out common warning spam
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*TensorFlow GPU support is not available.*')

# Resolve project root directory relative to this backend file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Compute path back up to Assignment2B project directory
assignment2b_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# Make Assignment2B and integration modules importable
sys.path.append(assignment2b_dir)
sys.path.append(os.path.join(assignment2b_dir, 'integration'))

# Import main TBRGS routing and ML integration entrypoint
from integration.main import run_tbrgs

# Create Flask app instance
app = Flask(__name__)
# Enable CORS so the React frontend on another port can call this API
CORS(app)

@app.route('/api/route', methods=['POST'])
def calculate_route():
    data = request.get_json()
    
    # Validate that mandatory origin and destination fields are present
    if not data or 'origin' not in data or 'destination' not in data:
        return jsonify({"error": "Please provide 'origin' and 'destination' SCATS nodes."}), 400
        
    # Read and normalise core request parameters from JSON body
    origin_input = str(data['origin'])
    destination_input = str(data['destination'])
    model_name = str(data.get('model', 'lstm')).lower()
    is_bidirectional = bool(data.get('bidirectional', False))

    # Map compact model ids plus bidirectional flag to explicit model names
    if model_name == 'lstm' and is_bidirectional:
        model_name = 'bidirectional_lstm'

    elif model_name == 'gru' and is_bidirectional:
        model_name = 'bidirectional_gru'
    
    elif model_name == 'custom_gcn_lstm':
        model_name = 'custom_gcn_lstm'

    # List of all model names that this backend understands
    supported_models = ['lstm', 'bidirectional_lstm', 'gru', 'bidirectional_gru', 'custom_gcn_lstm']
    if model_name not in supported_models:
        model_name = 'lstm'
    
    # Extract additional routing parameters falling back to configured defaults
    top_k = int(data.get('topK', DEFAULTS["topK"]))
    speed_limit = float(data.get('speedLimit', DEFAULTS["speedLimit"]))
    intersection_delay = float(data.get('intersectionDelay', DEFAULTS["intersectionDelay"]))
    depart_time = str(data.get('departTime', DEFAULTS["departTime"]))
    
    # Log summary of the incoming request to the server console
    print(f"\n[POST /api/route] New Request")
    print(f"  From SCATS: {origin_input} -> To SCATS: {destination_input}")
    print(f"  Model: {model_name} | topK: {top_k} | Speed Limit: {speed_limit}km/h | Delay: {intersection_delay}s | Depart: {depart_time}")

    # Use string forms directly as internal origin and destination ids
    origin = origin_input
    destination = destination_input
    # Path to static map coordinates file consumed by integration pipeline
    map_path = os.path.join(assignment2b_dir, 'map_data', 'boroodara_tbrgs_map_coordinates.txt')
    
    try:
        # Change working directory so integration code can use relative paths
        os.chdir(assignment2b_dir)
        
        # Run TBRGS pipeline to load ML model and compute top‑k routes
        start_time = time.perf_counter()
        routes = run_tbrgs(
            filename=map_path, 
            origin=origin, 
            destination=destination, 
            model_name=model_name,
            k_routes=top_k,
            speed_limit=speed_limit,
            intersection_delay=intersection_delay,
            depart_time=depart_time
        )
        elapsed_time = time.perf_counter() - start_time
        
        # Log basic stats about computed routes and timings
        print(f"  Found {len(routes)} routes in {elapsed_time:.3f}s:")
        for r in routes:
            nodes = len(r['path'])
            print(f"    - Route {r['route']} | {r['estimated_time_mins']} min | {r['distance_km']} km | {nodes} nodes")
        print("-" * 30)

        # Send successful response payload back to frontend
        return jsonify({
            "status": "success",
            "origin": origin,
            "destination": destination,
            "model": model_name,
            "findingTimeSeconds": round(elapsed_time, 3),
            "routes": routes
        }), 200
    except Exception as e:
        # Log full traceback and return generic error payload
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start development Flask server for the TBRGS backend API
    print("Starting TBRGS Machine Learning Backend API...")
    app.run(debug=True, port=5001)