"""
LLM Control API for MobiFlow-Auditor
Provides REST endpoints to query KPM and MobiFlow data
Location: MobiFlow-Auditor/src/llm_control_api.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import logging
from typing import Dict, List, Optional
import threading
import time

# These will be imported from the actual MobiFlow-Auditor modules
try:
    from manager.SdlManager import SdlManager
    from mobiflow.mobiflow import MobiFlow
except ImportError:
    # For standalone testing
    SdlManager = None
    MobiFlow = None

app = Flask(__name__)
CORS(app)
logger = logging.getLogger(__name__)

# Global SDL manager instance (set by main.py)
sdl_manager = None

def set_sdl_manager(manager):
    """Called by main.py to inject the SDL manager instance"""
    global sdl_manager
    sdl_manager = manager

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'mobiflow-auditor-llm-api',
        'timestamp': time.time()
    })

@app.route('/kpm/<ue_id>', methods=['GET'])
def get_kpm_data(ue_id: str):
    """
    Get KPM (Key Performance Metrics) data for a specific UE
    Returns: JSON with latest KPM indicators
    """
    try:
        if not sdl_manager:
            return jsonify({'error': 'SDL Manager not initialized'}), 503
        
        # Get KPM data from SDL
        kpm_data = sdl_manager.get_last_kpm_json(ue_id)
        
        if not kpm_data:
            return jsonify({'error': f'No KPM data found for UE {ue_id}'}), 404
        
        return jsonify({
            'ue_id': ue_id,
            'timestamp': time.time(),
            'kpm_data': kpm_data
        })
    
    except Exception as e:
        logger.error(f"Error fetching KPM data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/mobiflow/last', methods=['GET'])
def get_mobiflow_records():
    """
    Get last N MobiFlow records
    Query params: n (number of records, default 100)
    Returns: JSON array of MobiFlow records
    """
    try:
        if not sdl_manager:
            return jsonify({'error': 'SDL Manager not initialized'}), 503
        
        n = request.args.get('n', 100, type=int)
        n = min(n, 1000)  # Cap at 1000 records
        
        # Get MobiFlow records from SDL
        records = sdl_manager.get_mobiflow_json(n)
        
        return jsonify({
            'count': len(records),
            'timestamp': time.time(),
            'records': records
        })
    
    except Exception as e:
        logger.error(f"Error fetching MobiFlow records: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/mobiflow/ue/<ue_id>', methods=['GET'])
def get_ue_mobiflow(ue_id: str):
    """
    Get MobiFlow records for a specific UE
    Returns: JSON with UE-specific MobiFlow data including FBS indicators
    """
    try:
        if not sdl_manager:
            return jsonify({'error': 'SDL Manager not initialized'}), 503
        
        # Get UE-specific MobiFlow data
        ue_records = sdl_manager.get_ue_mobiflow_json(ue_id)
        
        if not ue_records:
            return jsonify({'error': f'No MobiFlow data found for UE {ue_id}'}), 404
        
        # Include FBS detection fields
        for record in ue_records:
            if 'suspected_fbs' not in record:
                record['suspected_fbs'] = False
            if 'attach_failures' not in record:
                record['attach_failures'] = 0
            if 'unauthenticated_id' not in record:
                record['unauthenticated_id'] = None
        
        return jsonify({
            'ue_id': ue_id,
            'timestamp': time.time(),
            'record_count': len(ue_records),
            'mobiflow_data': ue_records
        })
    
    except Exception as e:
        logger.error(f"Error fetching UE MobiFlow data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    """
    Get overall statistics from MobiFlow-Auditor
    Returns: JSON with system statistics
    """
    try:
        if not sdl_manager:
            return jsonify({'error': 'SDL Manager not initialized'}), 503
        
        stats = {
            'timestamp': time.time(),
            'total_ues': sdl_manager.get_ue_count(),
            'active_cells': sdl_manager.get_active_cells(),
            'total_records': sdl_manager.get_total_record_count(),
            'fbs_detections': sdl_manager.get_fbs_detection_count()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return jsonify({'error': str(e)}), 500

def run(host='0.0.0.0', port=8090, debug=False):
    """Run the Flask server"""
    logger.info(f"Starting LLM Control API on {host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)

def start_in_thread():
    """Start the API server in a background thread"""
    api_thread = threading.Thread(
        target=run,
        daemon=True,
        kwargs={'host': '0.0.0.0', 'port': 8090, 'debug': False}
    )
    api_thread.start()
    logger.info("LLM Control API started in background thread")
    return api_thread