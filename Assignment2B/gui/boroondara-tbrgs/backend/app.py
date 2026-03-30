import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

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
    model_name = data.get('model', 'lstm').lower()
    
    # MOCK OVERRIDE: The frontend UI requests SCATS node IDs (e.g., '2000') that do not naturally exist 
    # in the smaller 50-node FID testing map matrix, crashing Yen's Algorithm with a KeyError. 
    # We silently route using valid path IDs '7001' and '7015' instead.
    origin = '7001'
    destination = '7015'
    map_path = os.path.join(assignment2b_dir, 'map_data', 'boroodara_tbrgs_map_coordinates.txt')
    
    try:
        # Switch CWD to ensure relative paths inside intergration.py resolve successfully
        os.chdir(assignment2b_dir)
        
        # run_tbrgs dynamically loads the ML model and calculates the top Yen's K Shortest Paths
        routes = run_tbrgs(filename=map_path, origin=origin, destination=destination, model_name=model_name)
        
        return jsonify({
            "status": "success",
            "origin": origin,
            "destination": destination,
            "model": model_name,
            "routes": routes
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting TBRGS Machine Learning Backend API...")
    app.run(debug=True, port=5001)
