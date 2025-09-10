#!/usr/bin/env python3
"""
ArborNote Tree Detection Web Interface
MVP Flask application for tree detection using DeepForest
"""

import os
import sys
import json
import threading
import uuid
from datetime import datetime, timedelta
import tempfile
import zipfile
from pathlib import Path
import traceback
import time
import random
import csv
import shutil

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_file

# Optional imports with graceful fallbacks
try:
    from flask_socketio import SocketIO, emit
    HAS_SOCKETIO = True
except ImportError:
    print("Warning: Flask-SocketIO not available. Real-time updates disabled.")
    HAS_SOCKETIO = False

# Optional imports with graceful fallbacks
skip_deps = os.getenv('FORCE_SKIP_DEPENDENCIES', '').lower() in ('1', 'true', 'yes')

try:
    if not skip_deps:
        import pandas as pd
        HAS_PANDAS = True
    else:
        print("Skipping pandas due to FORCE_SKIP_DEPENDENCIES")
        HAS_PANDAS = False
except ImportError:
    print("Warning: pandas not available. CSV generation will use basic methods.")
    HAS_PANDAS = False

try:
    if not skip_deps:
        import numpy as np
        HAS_NUMPY = True
    else:
        print("Skipping numpy due to FORCE_SKIP_DEPENDENCIES")  
        HAS_NUMPY = False
except ImportError:
    print("Warning: numpy not available. Using Python random for simulations.")
    HAS_NUMPY = False

# Always import standard library modules for fallbacks
import random
import math
import sys
import os
import json
import uuid
import time
import threading
import traceback
import tempfile
import zipfile
import shutil
import csv

# Utility functions for numpy fallbacks
def safe_random_uniform(low, high):
    """Random uniform that works with or without numpy"""
    if HAS_NUMPY:
        return np.random.uniform(low, high)
    else:
        return random.uniform(low, high)

def safe_mean(values):
    """Mean calculation that works with or without numpy"""
    if not values:
        return 0
    if HAS_NUMPY:
        return np.mean(values)
    else:
        return sum(values) / len(values)

def safe_cos(x):
    """Cosine that works with or without numpy"""
    if HAS_NUMPY:
        return np.cos(x)
    else:
        return math.cos(x)

def safe_radians(x):
    """Radians conversion that works with or without numpy"""
    if HAS_NUMPY:
        return np.radians(x)
    else:
        return math.radians(x)

# Add PIL for image creation
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    print("Warning: PIL not available. Will create text placeholders instead of images.")
    HAS_PIL = False

try:
    from shapely.geometry import Polygon
    import pyproj
    HAS_GEO = True
except ImportError:
    print("Warning: Shapely/PyProj not available. Area calculation will use approximation.")
    HAS_GEO = False

# Import your existing modules (adjust paths as needed)
import sys
sys.path.append('..')

# Graceful imports for production - handle missing dependencies
try:
    if not skip_deps:
        from label_process.extract_nearmap import NearmapTileExtractor
        HAS_NEARMAP = True
    else:
        print("Skipping NearmapTileExtractor due to FORCE_SKIP_DEPENDENCIES")
        HAS_NEARMAP = False
except ImportError as e:
    print(f"Warning: Could not import NearmapTileExtractor: {e}")
    HAS_NEARMAP = False

try:
    if not skip_deps:
        from label_process.black_out import black_out_outside_geofence
        HAS_BLACKOUT = True
    else:
        print("Skipping black_out_outside_geofence due to FORCE_SKIP_DEPENDENCIES")
        HAS_BLACKOUT = False
except ImportError as e:
    print(f"Warning: Could not import black_out_outside_geofence: {e}")
    HAS_BLACKOUT = False

try:
    if not skip_deps:
        from deepforest import main as deepforest_main
        HAS_DEEPFOREST = True
    else:
        print("Skipping DeepForest due to FORCE_SKIP_DEPENDENCIES")
        HAS_DEEPFOREST = False
except ImportError as e:
    print(f"Warning: Could not import DeepForest: {e}")
    HAS_DEEPFOREST = False

# Import the new production pipeline
try:
    if not skip_deps:
        from label_process.production_pipeline import ProductionPipeline, create_production_config
        HAS_PRODUCTION_PIPELINE = True
    else:
        print("Skipping ProductionPipeline due to FORCE_SKIP_DEPENDENCIES")
        HAS_PRODUCTION_PIPELINE = False
except ImportError as e:
    print(f"Warning: Could not import ProductionPipeline: {e}")
    HAS_PRODUCTION_PIPELINE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arbornote-secret-key-2025'

# Initialize SocketIO only if available
if HAS_SOCKETIO:
    socketio = SocketIO(app, cors_allowed_origins="*")
else:
    socketio = None

# Load configuration
def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: config.json not found, using defaults")
        return {}

config = load_config()

# Configuration with fallbacks
CHECKPOINTS_DIR = os.getenv('CHECKPOINTS_DIR', config.get('paths', {}).get('checkpoints_dir', '../checkpoints'))
CHECKPOINTS_ALT_DIR = config.get('paths', {}).get('checkpoints_alt_dir', './checkpoints')

# Force simulation mode for dependency issues - enable real Nearmap extraction only
FORCE_SIMULATION_MODE = False  # Real Nearmap extraction enabled
FORCE_SKIP_DEPENDENCIES = True  # Skip problematic dependencies

AVAILABLE_MODELS = {
    'model_5' : 'model_epoch_5.ckpt',
    'model_15': 'model_epoch_15.ckpt',  # Default
    'model_25': 'model_epoch_25.ckpt',
    'model_35': 'model_epoch_35.ckpt',
    'model_45': 'model_epoch_45.ckpt',
    'model_55': 'model_epoch_55.ckpt',
    'model_65': 'model_epoch_65.ckpt',
}

DEFAULT_SETTINGS = config.get('default_settings', {
    'confidence_threshold': 0.25,
    'iou_threshold': 0.7,
    'containment_threshold': 0.75,
    'patch_size': 1000,
    'patch_overlap': 0.35,
    'enable_postprocessing': True,
    'target_mpp': 0.075
})

def create_placeholder_image(width, height, text, output_path):
    """Create a placeholder image with text"""
    if HAS_PIL:
        # Create actual image with PIL
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fall back to basic if needed
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Draw text on image
        text_lines = text.split('\n')
        y_offset = height // 2 - (len(text_lines) * 15)
        
        for line in text_lines:
            if font:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x_offset = (width - text_width) // 2
                draw.text((x_offset, y_offset), line, fill='black', font=font)
            else:
                draw.text((10, y_offset), line, fill='black')
            y_offset += 25
        
        img.save(output_path)
        print(f"✓ Created placeholder image: {output_path}")
    else:
        # Fall back to text file
        with open(output_path, 'w') as f:
            f.write(f"Placeholder Image\n{text}")
        print(f"✓ Created placeholder text file: {output_path}")

def create_visualization_image(predictions, geofence, output_path, area_acres, processed_image_path=None):
    """Create a visualization image with tree predictions and bounding boxes similar to validate_model.py"""
    if not HAS_PIL:
        # Fall back to creating a text summary
        with open(output_path, 'w') as f:
            f.write(f"Tree Detection Visualization\n")
            f.write(f"Area: {area_acres} acres\n")
            f.write(f"Trees detected: {len(predictions)}\n\n")
            for i, pred in enumerate(predictions[:10]):  # Show first 10
                f.write(f"Tree {i+1}: {pred['latitude']:.6f}, {pred['longitude']:.6f} (conf: {pred['confidence']:.2f})\n")
        print(f"✓ Created visualization text file: {output_path}")
        return
    
    # Try to use the actual processed image as background if available
    base_image = None
    if processed_image_path and os.path.exists(processed_image_path):
        try:
            base_image = Image.open(processed_image_path)
            # Convert to RGB if needed
            if base_image.mode != 'RGB':
                base_image = base_image.convert('RGB')
            img_width, img_height = base_image.size
            print(f"Using processed image as background: {img_width}x{img_height}")
        except Exception as e:
            print(f"Could not load processed image: {e}, using default background")
            base_image = None
    
    # Create base image if we don't have a real one
    if base_image is None:
        img_width, img_height = 800, 600
        base_image = Image.new('RGB', (img_width, img_height), color='darkgreen')
        img_width, img_height = base_image.size
    
    # Create a copy for drawing
    img = base_image.copy()
    
    # Use opencv-style drawing if we can import it, otherwise use PIL
    try:
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Draw predictions with OpenCV (similar to validate_model.py)
        for i, pred in enumerate(predictions):
            # Map lat/lon to pixel coordinates
            # For now, use simple mapping - in a real implementation, you'd use the image metadata
            if 'bbox_pixels' in pred:
                # Use pixel coordinates if available
                x1 = int(pred['bbox_pixels']['xmin'])
                y1 = int(pred['bbox_pixels']['ymin']) 
                x2 = int(pred['bbox_pixels']['xmax'])
                y2 = int(pred['bbox_pixels']['ymax'])
            else:
                # Map from lat/lon to approximate pixel position
                # This is a simple mapping - real implementation would use image metadata
                center_x = int((pred.get('longitude', 0) + 180) / 360 * img_width)
                center_y = int((90 - pred.get('latitude', 0)) / 180 * img_height)
                bbox_size = 15  # Default box size
                x1, y1 = center_x - bbox_size, center_y - bbox_size
                x2, y2 = center_x + bbox_size, center_y + bbox_size
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width-1))
            y1 = max(0, min(y1, img_height-1)) 
            x2 = max(0, min(x2, img_width-1))
            y2 = max(0, min(y2, img_height-1))
            
            confidence = pred.get('confidence', 0.5)
            
            # Color coding: Green for high confidence (>0.7), Yellow for medium (0.4-0.7), Red for low (<0.4)
            if confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.4:
                color = (0, 255, 255)  # Yellow - medium confidence  
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            conf_text = f"{confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            
            # Position text above the box
            text_y = max(y1 - 10, 15)
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
            
            # Draw semi-transparent background for text
            overlay = img_cv.copy()
            cv2.rectangle(overlay, (x1 - 2, text_y - text_height - 2), 
                         (x1 + text_width + 2, text_y + baseline + 2), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, img_cv, 0.3, 0, img_cv)
            
            # Draw the confidence text
            cv2.putText(img_cv, conf_text, (x1, text_y), font, font_scale, (0, 0, 0), font_thickness)
        
        # Add title and statistics
        title = f"Tree Detection Results - {area_acres} acres"
        stats_text = f"Trees: {len(predictions)} | Avg Confidence: {safe_mean([p['confidence'] for p in predictions]) * 100:.1f}%"
        
        # Draw title background
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.7
        (title_w, title_h), _ = cv2.getTextSize(title, title_font, title_scale, 2)
        cv2.rectangle(img_cv, (10, 10), (20 + title_w, 30 + title_h), (0, 0, 0), -1)
        cv2.putText(img_cv, title, (15, 25 + title_h), title_font, title_scale, (255, 255, 255), 2)
        
        # Draw stats background
        (stats_w, stats_h), _ = cv2.getTextSize(stats_text, title_font, 0.5, 1)
        cv2.rectangle(img_cv, (10, img_height - 30), (20 + stats_w, img_height - 5), (0, 0, 0), -1)
        cv2.putText(img_cv, stats_text, (15, img_height - 10), title_font, 0.5, (255, 255, 255), 1)
        
        # Convert back to PIL and save
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
    except ImportError:
        # Fall back to PIL drawing if OpenCV not available
        draw = ImageDraw.Draw(img)
        
        # Draw predictions with PIL
        for i, pred in enumerate(predictions):
            # Map coordinates similar to above
            if 'bbox_pixels' in pred:
                x1 = int(pred['bbox_pixels']['xmin'])
                y1 = int(pred['bbox_pixels']['ymin'])
                x2 = int(pred['bbox_pixels']['xmax']) 
                y2 = int(pred['bbox_pixels']['ymax'])
            else:
                center_x = int((pred.get('longitude', 0) + 180) / 360 * img_width)
                center_y = int((90 - pred.get('latitude', 0)) / 180 * img_height)
                bbox_size = 15
                x1, y1 = center_x - bbox_size, center_y - bbox_size
                x2, y2 = center_x + bbox_size, center_y + bbox_size
            
            confidence = pred.get('confidence', 0.5)
            
            # Color coding for PIL (RGB format)
            if confidence > 0.7:
                color = 'green'  # High confidence
            elif confidence > 0.4:
                color = 'yellow'  # Medium confidence
            else:
                color = 'red'  # Low confidence
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw confidence text
            if i < 50:  # Limit labels to avoid clutter
                conf_text = f"{confidence:.2f}"
                draw.text((x1, max(y1-15, 0)), conf_text, fill='white')
        
        # Draw title and stats
        try:
            title_font = ImageFont.truetype("arial.ttf", 16)
            stats_font = ImageFont.truetype("arial.ttf", 12)
        except:
            title_font = stats_font = None
        
        title = f"Tree Detection Results - {area_acres} acres"
        draw.text((10, 10), title, fill='white', font=title_font)
        
        stats_text = f"Trees: {len(predictions)} | Avg Confidence: {safe_mean([p['confidence'] for p in predictions]) * 100:.1f}%"
        draw.text((10, img_height - 25), stats_text, fill='white', font=stats_font)
    
    # Save the final image
    img.save(output_path)
    print(f"✓ Created detailed visualization with bounding boxes: {output_path}")

# Global job storage
active_jobs = {}

@app.route('/health')
def health_check():
    """Health check endpoint for production monitoring"""
    
    # Check system status
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0',
        'environment': os.getenv('FLASK_ENV', 'development'),
        'components': {}
    }
    
    # Check Flask
    health_status['components']['flask'] = {'status': 'up'}
    
    # Check SocketIO
    health_status['components']['socketio'] = {
        'status': 'up' if HAS_SOCKETIO else 'degraded',
        'message': 'Available' if HAS_SOCKETIO else 'Using polling fallback'
    }
    
    # Check geospatial libraries
    health_status['components']['geospatial'] = {
        'status': 'up' if HAS_GEO else 'degraded',
        'message': 'Full accuracy' if HAS_GEO else 'Approximate calculations'
    }
    
    # Check Nearmap integration
    try:
        from label_process.extract_nearmap import NearmapTileExtractor
        api_key = os.getenv('NEARMAP_API_KEY')
        nearmap_status = 'up' if api_key else 'degraded'
        nearmap_message = 'Configured' if api_key else 'API key missing - simulation mode'
    except ImportError:
        nearmap_status = 'degraded'
        nearmap_message = 'Module not available - simulation mode'
    
    health_status['components']['nearmap'] = {
        'status': nearmap_status,
        'message': nearmap_message
    }
    
    # Check DeepForest
    try:
        import deepforest
        deepforest_status = 'up'
        deepforest_message = 'Available'
    except ImportError:
        deepforest_status = 'degraded'
        deepforest_message = 'Not available - simulation mode'
    
    health_status['components']['deepforest'] = {
        'status': deepforest_status,
        'message': deepforest_message
    }
    
    # Check Production Pipeline
    try:
        from label_process.production_pipeline import ProductionPipeline
        pipeline_status = 'up'
        pipeline_message = 'Production pipeline available'
    except ImportError:
        pipeline_status = 'degraded'
        pipeline_message = 'Production pipeline not available - using legacy processing'
    
    health_status['components']['production_pipeline'] = {
        'status': pipeline_status,
        'message': pipeline_message
    }
    
    # Check model files
    models_available = []
    for model_name, filename in AVAILABLE_MODELS.items():
        model_path = find_model_path(filename)
        if model_path:
            models_available.append(model_name)
    
    health_status['components']['models'] = {
        'status': 'up' if models_available else 'degraded',
        'available_models': models_available,
        'message': f'{len(models_available)} models available' if models_available else 'No models found'
    }
    
    # Check directories
    required_dirs = ['temp_files', 'logs', 'outputs']
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    health_status['components']['filesystem'] = {
        'status': 'up' if not missing_dirs else 'degraded',
        'missing_directories': missing_dirs
    }
    
    # Overall status
    component_statuses = [comp['status'] for comp in health_status['components'].values()]
    if any(status == 'down' for status in component_statuses):
        health_status['status'] = 'unhealthy'
    elif any(status == 'degraded' for status in component_statuses):
        health_status['status'] = 'degraded'
    
    # Set HTTP status code based on health
    status_code = 200 if health_status['status'] in ['healthy', 'degraded'] else 503
    
    return jsonify(health_status), status_code

@app.route('/')
def index():
    """Main interface page"""
    return render_template('index.html')

def find_model_path(filename):
    """Find model file in multiple possible locations"""
    possible_paths = [
        os.path.join(CHECKPOINTS_DIR, filename),
        os.path.join(CHECKPOINTS_ALT_DIR, filename),
        filename  # Current directory as last resort
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

@app.route('/api/models')
def get_available_models():
    """Get list of available model checkpoints"""
    models = []
    for key, filename in AVAILABLE_MODELS.items():
        model_path = find_model_path(filename)
        models.append({
            'id': key,
            'name': key.replace('_', ' ').title(),
            'filename': filename,
            'path': model_path,
            'exists': model_path is not None,
            'default': key == 'model_55'
        })
    return jsonify(models)

@app.route('/api/dates')
def get_available_dates():
    """Get available imagery dates (Aug-Oct, past 3 years)"""
    current_year = datetime.now().year
    dates = []
    
    for year in range(current_year - 3, current_year + 1):
        for month in [8, 9, 10]:  # August, September, October
            # Only include past dates
            date = datetime(year, month, 1)
            if date <= datetime.now():
                dates.append({
                    'value': f"{year}-{month:02d}",
                    'label': date.strftime('%B %Y'),
                    'year': year,
                    'month': month
                })
    
    return jsonify(sorted(dates, key=lambda x: x['value'], reverse=True))

@app.route('/api/calculate-area', methods=['POST'])
def calculate_area():
    """Calculate area and cost from geofence coordinates"""
    try:
        data = request.json
        geofence = data.get('geofence', [])
        
        if len(geofence) < 3:
            return jsonify({'error': 'Need at least 3 points for geofence'}), 400
        
        area_info = calculate_area_and_cost(geofence)
        return jsonify(area_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def start_processing():
    """Start the tree detection process"""
    try:
        data = request.json
        
        # Validate input
        geofence = data.get('geofence', [])
        if len(geofence) < 3:
            return jsonify({'error': 'Invalid geofence coordinates'}), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Calculate area and cost
        area_info = calculate_area_and_cost(geofence)
        
        # Store job data
        active_jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'message': 'Processing queued...',
            'geofence': geofence,
            'settings': {**DEFAULT_SETTINGS, **data.get('settings', {})},
            'area_info': area_info,
            'imagery_date': data.get('imagery_date'),
            'model': data.get('model', 'model_55'),
            'created_at': datetime.now().isoformat(),
            'files': {},
            'results': {}
        }
        
        # Start background processing
        thread = threading.Thread(target=process_job_async, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'area_info': area_info,
            'estimated_time_minutes': estimate_processing_time(area_info['acres'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    """Get current job status"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(active_jobs[job_id])

@app.route('/api/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    """Download specific file from job"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    file_path = job['files'].get(file_type)
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Set appropriate mimetype for CSV files
    mimetype = None
    if file_path.endswith('.csv'):
        mimetype = 'text/csv'
    elif file_path.endswith('.kml'):
        mimetype = 'application/vnd.google-earth.kml+xml'
    elif file_path.endswith('.json'):
        mimetype = 'application/json'
    
    return send_file(file_path, as_attachment=True, mimetype=mimetype)

@app.route('/api/download-all/<job_id>')
def download_all_files(job_id):
    """Download all files as ZIP archive"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    
    # Create temporary ZIP file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, 'w') as zip_file:
            for file_type, file_path in job['files'].items():
                if os.path.exists(file_path):
                    zip_file.write(file_path, f"{file_type}_{os.path.basename(file_path)}")
        
        return send_file(tmp_zip.name, as_attachment=True, 
                        download_name=f"tree_detection_{job_id}.zip")

def calculate_area_and_cost(geofence_coords):
    """Calculate area in acres and total cost"""
    if HAS_GEO:
        # Use accurate projection-based calculation
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        projected_coords = [transformer.transform(lon, lat) for lon, lat in geofence_coords]
        
        polygon = Polygon(projected_coords)
        area_sq_meters = polygon.area
    else:
        # Use approximate calculation (Shoelace formula with rough conversion)
        print("Using approximate area calculation (install Shapely for accurate results)")
        
        # Simple polygon area calculation using Shoelace formula
        n = len(geofence_coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += geofence_coords[i][0] * geofence_coords[j][1]
            area -= geofence_coords[j][0] * geofence_coords[i][1]
        area = abs(area) / 2.0
        
        # Rough conversion from decimal degrees to square meters (very approximate!)
        # This is only for demo purposes - real deployment should use proper projections
        lat_avg = sum(coord[1] for coord in geofence_coords) / len(geofence_coords)
        meters_per_degree = 111320 * abs(safe_cos(safe_radians(lat_avg)))
        area_sq_meters = area * (meters_per_degree ** 2)
    
    area_acres = area_sq_meters / 4047  # Convert to acres
    cost = area_acres * 500  # $500 per acre
    
    return {
        'acres': round(area_acres, 3),
        'square_meters': round(area_sq_meters, 2),
        'cost_usd': round(cost, 2)
    }

def estimate_processing_time(acres):
    """Estimate processing time based on area"""
    # Rough estimates: ~2-3 minutes per acre
    base_time = acres * 2.5
    return max(1, round(base_time))

def process_job_async(job_id):
    """Background processing function"""
    job = active_jobs[job_id]
    
    try:
        # Step 1: Extract Nearmap imagery
        update_job_progress(job_id, 'processing', 10, 'Extracting Nearmap imagery...')
        extract_nearmap_step(job)
        
        # Step 2: Process and normalize image
        update_job_progress(job_id, 'processing', 30, 'Processing image (normalize & mask)...')
        process_image_step(job)
        
        # Step 3: Run AI model
        update_job_progress(job_id, 'processing', 60, 'Running DeepForest AI model...')
        run_model_step(job)
        
        # Step 4: Post-process and generate outputs
        update_job_progress(job_id, 'processing', 85, 'Post-processing results...')
        generate_outputs_step(job)
        
        # Complete
        update_job_progress(job_id, 'completed', 100, 'Tree detection complete!')
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"Job {job_id} error: {error_msg}")
        print(traceback.format_exc())
        update_job_progress(job_id, 'error', 0, error_msg)

def update_job_progress(job_id, status, progress, message):
    """Update job progress and notify clients"""
    if job_id in active_jobs:
        active_jobs[job_id]['status'] = status
        active_jobs[job_id]['progress'] = progress
        active_jobs[job_id]['message'] = message
        
        # Emit progress update via WebSocket if available
        if HAS_SOCKETIO and socketio:
            socketio.emit('job_progress', {
                'job_id': job_id,
                'status': status,
                'progress': progress,
                'message': message
            })

def extract_nearmap_step(job):
    """Step 1: Extract Nearmap imagery"""
    import time
    
    if not HAS_NEARMAP or FORCE_SIMULATION_MODE:
        # Simulation mode - create mock imagery data
        print(f"Running Nearmap extraction in simulation mode for job {job['id']}")
        time.sleep(2)  # Simulate API call time
        
        # Create temp directory for this job
        job_temp_dir = os.path.join('temp_files', job['id'])
        os.makedirs(job_temp_dir, exist_ok=True)
        
        # Create mock file paths
        job['files']['raw_image'] = os.path.join(job_temp_dir, 'simulated_nearmap_image.jpg')
        job['files']['metadata'] = os.path.join(job_temp_dir, 'image_metadata.json')
        
        # Create mock metadata
        geofence = job['geofence']
        mock_metadata = {
            'bounds': {
                'min_lat': min(coord[1] for coord in geofence),
                'max_lat': max(coord[1] for coord in geofence),
                'min_lon': min(coord[0] for coord in geofence),
                'max_lon': max(coord[0] for coord in geofence)
            },
            'area_acres': job['area_info']['acres'],
            'resolution_mpp': 0.075,
            'simulation_mode': True,
            'api_key_used': 'SIMULATION',
            'extraction_date': datetime.now().isoformat()
        }
        
        # Save mock metadata
        with open(job['files']['metadata'], 'w') as f:
            json.dump(mock_metadata, f, indent=2)
        
        # Create a realistic placeholder image (800x600)
        image_text = f"Simulated Nearmap Imagery\nArea: {job['area_info']['acres']} acres\nResolution: 0.075 MPP\nExtraction Date: {datetime.now().strftime('%Y-%m-%d')}"
        create_placeholder_image(800, 600, image_text, job['files']['raw_image'])
        
        print(f"✓ Simulated Nearmap extraction completed for {job['area_info']['acres']} acres")
        return
    
    # Real Nearmap extraction using production modules
    print(f"Using real NearmapTileExtractor for job {job['id']}")
    
    try:
        # Create temp directory for this job
        job_temp_dir = os.path.join('temp_files', job['id'])
        os.makedirs(job_temp_dir, exist_ok=True)
        
        # Create KML file from geofence
        kml_path = os.path.join(job_temp_dir, 'geofence_boundary.kml')
        create_kml_from_geofence(job['geofence'], kml_path)
        
        # Initialize Nearmap extractor
        from label_process.extract_nearmap import NearmapTileExtractor
        extractor = NearmapTileExtractor(
            api_key=os.getenv('NEARMAP_API_KEY'),
            target_mpp=job['settings'].get('target_mpp', 0.075)
        )
        
        # Extract imagery with date range instead of specific date
        imagery_date = job.get('imagery_date')
        
        if imagery_date:
            print(f"Using specified date: {imagery_date}")
        else:
            # Use date range approach - let Nearmap return best available imagery in the past year
            print("No date specified, requesting imagery from past year...")
            from datetime import datetime, timedelta
            
            # Request imagery from 1 year ago until today - let Nearmap find the best available
            today = datetime.now()
            one_year_ago = today - timedelta(days=365)
            
            # Format as date range for Nearmap API
            imagery_date = f"since={one_year_ago.strftime('%Y-%m-%d')}&until={today.strftime('%Y-%m-%d')}"
            print(f"Using date range: {imagery_date}")
        
        # Run the extraction with fallback date logic
        location_name = f"job_{job['id'][:8]}"
        
        # Try the extraction with automatic date fallback built into the extractor
        results = extractor.extract_region(
            kml_path=kml_path,
            date=imagery_date,  # The extractor will find the closest available date
            output_dir=job_temp_dir,
            location_name=location_name
        )
        
        # Set file paths from extraction results
        job['files']['raw_image'] = results.get('image_path', os.path.join(job_temp_dir, f'{location_name}_highres.jpg'))
        job['files']['metadata'] = results.get('metadata_path', os.path.join(job_temp_dir, f'{location_name}_highres_metadata.json'))
        job['files']['geofence_kml'] = kml_path
        
        # Log the actual imagery date information
        actual_date = results.get('date_actual')
        requested_date = results.get('date_requested', imagery_date)
        if actual_date and actual_date != requested_date:
            print(f"✓ Real Nearmap extraction completed - Requested: {requested_date}, Actual: {actual_date}")
        else:
            print(f"✓ Real Nearmap extraction completed for {job['area_info']['acres']} acres")
            
        # Store date information in job metadata for later reference
        job['imagery_info'] = {
            'date_requested': requested_date,
            'date_actual': actual_date,
            'survey_ids': results.get('survey_ids', []),
            'capture_dates': results.get('capture_dates', []),
            'zoom_level': results.get('zoom'),
            'resolution_actual': results.get('actual_geographic_mpp')
        }
        
    except Exception as e:
        print(f"Error in real Nearmap extraction: {e}")
        print("Falling back to simulation mode...")
        
        # Fall back to simulation mode if real extraction fails
        job['files']['raw_image'] = os.path.join(job_temp_dir, 'simulated_nearmap_image.jpg')
        job['files']['metadata'] = os.path.join(job_temp_dir, 'image_metadata.json')
        
        # Create fallback files
        mock_metadata = {
            'bounds': {
                'min_lat': min(coord[1] for coord in job['geofence']),
                'max_lat': max(coord[1] for coord in job['geofence']),
                'min_lon': min(coord[0] for coord in job['geofence']),
                'max_lon': max(coord[0] for coord in job['geofence'])
            },
            'area_acres': job['area_info']['acres'],
            'resolution_mpp': 0.075,
            'simulation_mode': True,
            'fallback_reason': str(e),
            'extraction_date': datetime.now().isoformat()
        }
        
        with open(job['files']['metadata'], 'w') as f:
            json.dump(mock_metadata, f, indent=2)
        
        image_text = f"Simulated Nearmap Imagery\n(Fallback Mode)\nArea: {job['area_info']['acres']} acres\nReason: {str(e)[:50]}..."
        create_placeholder_image(800, 600, image_text, job['files']['raw_image'])

def process_image_step(job):
    """Step 2: Process and normalize image"""
    import time
    
    # Check if we have real imagery from the previous step
    real_imagery_available = (
        'raw_image' in job['files'] and 
        job['files']['raw_image'] and 
        os.path.exists(job['files']['raw_image']) and 
        job['files']['raw_image'].endswith('.jpg')  # Real image, not text placeholder
    )
    
    if not HAS_BLACKOUT or FORCE_SIMULATION_MODE:
        if real_imagery_available:
            # We have real imagery but no black_out module - use the raw image with resolution normalization
            print(f"Using real imagery with resolution normalization for job {job['id']}")
            print("Note: Install 'opencv-python' and 'pykml' for full geofence masking functionality")
            
            job_temp_dir = os.path.join('temp_files', job['id'])
            
            # Load metadata to check current resolution
            try:
                with open(job['files']['metadata'], 'r') as f:
                    metadata = json.load(f)
                
                current_mpp = metadata.get('resolution_mpp', metadata.get('actual_resolution_meters_per_pixel', 0.062))
                target_mpp = job['settings'].get('target_mpp', 0.075)
                scale_factor = current_mpp / target_mpp
                
                print(f"Current resolution: {current_mpp:.6f} m/px")
                print(f"Target resolution: {target_mpp:.6f} m/px")
                print(f"Scale factor: {scale_factor:.4f}")
                print(f"Scale difference from 1.0: {abs(scale_factor - 1.0):.4f}")
                
                raw_image_path = job['files']['raw_image']
                processed_image_path = os.path.join(job_temp_dir, 'processed_image.jpg')
                
                # Only resample if there's a significant difference (>5%)
                if abs(scale_factor - 1.0) > 0.05:
                    print(f"✓ Resampling required: {scale_factor:.4f}x scaling (threshold: 0.05)")
                    resample_image_to_target_resolution(
                        raw_image_path,
                        processed_image_path,
                        scale_factor,
                        target_mpp
                    )
                else:
                    print(f"✓ Resolution already close to target, copying without resampling")
                    import shutil
                    shutil.copy2(raw_image_path, processed_image_path)
                
                job['files']['processed_image'] = processed_image_path
                
            except Exception as e:
                print(f"Error during resolution normalization: {e}")
                print("Falling back to copying original image...")
                raw_image_path = job['files']['raw_image']
                processed_image_path = os.path.join(job_temp_dir, 'processed_image.jpg')
                import shutil
                shutil.copy2(raw_image_path, processed_image_path)
                job['files']['processed_image'] = processed_image_path
            
            # Create a simple mask placeholder
            job['files']['geofence_mask'] = os.path.join(job_temp_dir, 'geofence_mask_placeholder.txt')
            with open(job['files']['geofence_mask'], 'w') as f:
                f.write("Geofence mask placeholder - install opencv-python and pykml for full masking functionality")
            
            print(f"✓ Real image processing with resolution normalization completed")
            return
        else:
            # No real imagery - create simulation placeholders
            print(f"Running image processing in simulation mode for job {job['id']}")
            time.sleep(1)  # Simulate processing time
            
            job_temp_dir = os.path.join('temp_files', job['id'])
            job['files']['processed_image'] = os.path.join(job_temp_dir, 'simulated_processed_image.jpg')
            job['files']['geofence_mask'] = os.path.join(job_temp_dir, 'geofence_mask.png')
            
            # Create processed image placeholder
            processed_text = f"Processed Imagery\nMasked to Geofence\nArea: {job['area_info']['acres']} acres"
            create_placeholder_image(800, 600, processed_text, job['files']['processed_image'])
            
            # Create mask image placeholder  
            mask_text = f"Geofence Mask\nShowing boundary area"
            create_placeholder_image(800, 600, mask_text, job['files']['geofence_mask'])
            
            print(f"✓ Simulated image processing completed")
            return
    
    # Real processing using production black_out module
    print(f"Using real black_out_outside_geofence for job {job['id']}")
    
    try:
        job_temp_dir = os.path.join('temp_files', job['id'])
        
        # Set up file paths
        raw_image_path = job['files']['raw_image']
        metadata_path = job['files']['metadata']
        kml_path = job['files'].get('geofence_kml')
        
        # STEP 1: RESOLUTION NORMALIZATION
        # Load metadata to check current resolution and normalize if needed
        normalized_image_path = os.path.join(job_temp_dir, 'normalized_image.jpg')
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            current_mpp = metadata.get('resolution_mpp', metadata.get('actual_resolution_meters_per_pixel', 0.062))
            target_mpp = job['settings'].get('target_mpp', 0.075)
            scale_factor = current_mpp / target_mpp
            
            print(f"🔍 RESOLUTION NORMALIZATION:")
            print(f"   Current resolution: {current_mpp:.6f} m/px")
            print(f"   Target resolution: {target_mpp:.6f} m/px")
            print(f"   Scale factor: {scale_factor:.4f}")
            print(f"   Scale difference from 1.0: {abs(scale_factor - 1.0):.4f}")
            
            # Only resample if there's a significant difference (>5%)
            if abs(scale_factor - 1.0) > 0.05:
                print(f"   ✓ Resampling required: {scale_factor:.4f}x scaling (threshold: 0.05)")
                resample_image_to_target_resolution(
                    raw_image_path,
                    normalized_image_path,
                    scale_factor,
                    target_mpp
                )
                
                # Create updated metadata for the normalized image
                normalized_metadata_path = os.path.join(job_temp_dir, 'normalized_metadata.json')
                create_normalized_metadata(metadata_path, normalized_metadata_path, scale_factor, target_mpp)
                
                input_for_masking = normalized_image_path
                metadata_for_masking = normalized_metadata_path
                job['files']['normalized_metadata'] = normalized_metadata_path  # Store for later use
                print(f"   ✓ Normalized image created: {normalized_image_path}")
                print(f"   ✓ Normalized metadata created: {normalized_metadata_path}")
            else:
                print(f"   ✓ Resolution already close to target, no resampling needed")
                input_for_masking = raw_image_path
                metadata_for_masking = metadata_path
                
        except Exception as e:
            print(f"   ❌ Error during resolution normalization: {e}")
            print(f"   ✓ Falling back to original image...")
            input_for_masking = raw_image_path
            metadata_for_masking = metadata_path
        
        # STEP 2: GEOFENCE MASKING
        # If no KML file exists, create one
        if not kml_path or not os.path.exists(kml_path):
            kml_path = os.path.join(job_temp_dir, 'geofence_boundary.kml')
            create_kml_from_geofence(job['geofence'], kml_path)
            job['files']['geofence_kml'] = kml_path
        
        # Run the real geofence processing on the normalized image
        from label_process.black_out import black_out_outside_geofence
        
        final_processed_path = os.path.join(job_temp_dir, 'processed_image.jpg')
        
        print(f"🎭 GEOFENCE MASKING:")
        print(f"   Input image: {input_for_masking}")
        print(f"   Input metadata: {metadata_for_masking}")
        print(f"   Output image: {final_processed_path}")
        
        black_out_outside_geofence(
            image_path=input_for_masking,
            metadata_path=metadata_for_masking,
            kml_path=kml_path,
            output_path=final_processed_path,
            save_mask=True,
            labels_path=None  # No ignore boxes for now
        )
        
        # Update the processed image path to point to the final masked version
        job['files']['processed_image'] = final_processed_path
        
        # Update the processed image path to point to the final masked version
        job['files']['processed_image'] = final_processed_path
        
        # Check if mask was created
        mask_path = final_processed_path.replace('.jpg', '_mask.png')
        if os.path.exists(mask_path):
            job['files']['geofence_mask'] = mask_path
        
        print(f"✓ Real image processing completed")
        
    except Exception as e:
        print(f"Error in real image processing: {e}")
        print("Falling back to simulation mode...")
        
        # Fall back to simulation mode if processing fails
        job['files']['processed_image'] = os.path.join(job_temp_dir, 'simulated_processed_image.jpg')
        job['files']['geofence_mask'] = os.path.join(job_temp_dir, 'geofence_mask.png')
        
        # Create fallback processed image
        processed_text = f"Processed Imagery\n(Fallback Mode)\nArea: {job['area_info']['acres']} acres\nError: {str(e)[:50]}..."
        create_placeholder_image(800, 600, processed_text, job['files']['processed_image'])
        
        # Create fallback mask
        mask_text = f"Geofence Mask\n(Fallback Mode)"
        create_placeholder_image(800, 600, mask_text, job['files']['geofence_mask'])

def run_model_step(job):
    """Step 3: Run DeepForest model using production pipeline"""
    import time
    
    update_job_progress(job['id'], 'processing', 60, 'Running AI tree detection...')
    
    job_temp_dir = os.path.join('temp_files', job['id'])
    
    # Try to use the new production pipeline first
    if HAS_PRODUCTION_PIPELINE and not FORCE_SIMULATION_MODE:
        print(f"Using ProductionPipeline for job {job['id']}")
        
        try:
            # Create production configuration from job settings
            config = create_production_config(
                confidence_threshold=job['settings'].get('confidence_threshold', 0.25),
                iou_threshold=job['settings'].get('iou_threshold', 0.7),
                patch_size=job['settings'].get('patch_size', 1000),
                patch_overlap=job['settings'].get('patch_overlap', 0.35),
                containment_threshold=job['settings'].get('containment_threshold', 0.75),
                enable_postprocessing=job['settings'].get('enable_postprocessing', True),
                enable_adaptive_filtering=True,
                show_removed_boxes=True,
                save_csv=True,
                save_summary=True
            )
            
            # Create pipeline
            pipeline = ProductionPipeline(config)
            
            # Get model path
            model_filename = AVAILABLE_MODELS.get(job['model'], 'model_epoch_55.ckpt')
            model_path = find_model_path(model_filename)
            
            if not model_path:
                raise FileNotFoundError(f"Model not found: {model_filename}")
            
            # Prepare paths
            image_path = job['files']['processed_image']
            geofence_kml_path = job['files'].get('geofence_kml')  # Optional
            metadata_path = job['files'].get('normalized_metadata') or job['files'].get('metadata')  # Optional
            
            # Run the complete pipeline
            results = pipeline.run_complete_pipeline(
                image_path=image_path,
                model_path=model_path,
                output_dir=job_temp_dir,
                kml_path=geofence_kml_path,
                metadata_path=metadata_path
            )
            
            if results['success']:
                # Convert pipeline predictions to our format
                flask_predictions = []
                for pred in results['predictions']:
                    if HAS_GEO:  # If Shapely is available
                        center_x = pred['center'].x
                        center_y = pred['center'].y
                        bounds = pred['box'].bounds
                    else:
                        center_x = pred['center']['x']
                        center_y = pred['center']['y']
                        bounds = [pred['box']['xmin'], pred['box']['ymin'], pred['box']['xmax'], pred['box']['ymax']]
                    
                    # Convert pixel coordinates to lat/lon if metadata is available
                    lat, lon = 0.0, 0.0
                    if metadata_path and os.path.exists(metadata_path):
                        lat, lon = pipeline.pixel_to_latlon(center_x, center_y)
                    
                    flask_predictions.append({
                        'longitude': lon,
                        'latitude': lat,
                        'confidence': pred['confidence'],
                        'pixel_x': center_x,
                        'pixel_y': center_y,
                        'bbox': {
                            'xmin': bounds[0],
                            'ymin': bounds[1], 
                            'xmax': bounds[2],
                            'ymax': bounds[3]
                        }
                    })
                
                # Store results in job
                job['results']['predictions'] = flask_predictions
                job['results']['raw_prediction_count'] = results['results'].initial_predictions
                job['results']['filtered_prediction_count'] = len(flask_predictions)
                job['results']['model_used'] = f"{job['model']} (PRODUCTION_PIPELINE)"
                job['results']['pipeline_summary'] = {
                    'initial_predictions': results['results'].initial_predictions,
                    'confidence_filtered': results['results'].confidence_filtered,
                    'geofence_filtered': results['results'].geofence_filtered,
                    'adaptive_filtered': results['results'].adaptive_filtered,
                    'postprocessed': results['results'].postprocessed,
                    'adaptive_threshold': results['results'].adaptive_threshold,
                    'processing_time': results['results'].processing_time
                }
                
                # Update job files with pipeline outputs
                if 'visualization' in results['output_files']:
                    job['files']['visualization_detailed'] = results['output_files']['visualization']
                if 'csv' in results['output_files']:
                    job['files']['predictions_csv_detailed'] = results['output_files']['csv']
                if 'summary' in results['output_files']:
                    job['files']['pipeline_summary'] = results['output_files']['summary']
                
                print(f"✓ ProductionPipeline completed: {len(flask_predictions)} final predictions")
                return
                
            else:
                print(f"ProductionPipeline failed: {results.get('error', 'Unknown error')}")
                raise Exception(f"Pipeline failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"ProductionPipeline error: {e}")
            print("Falling back to legacy processing...")
    
    # Legacy processing (simulation or basic DeepForest)
    if not HAS_DEEPFOREST or FORCE_SIMULATION_MODE:
        # Check if we have real processed imagery
        real_imagery_available = (
            'processed_image' in job['files'] and 
            job['files']['processed_image'] and 
            os.path.exists(job['files']['processed_image']) and 
            job['files']['processed_image'].endswith('.jpg')  # Real image, not text placeholder
        )
        
        if real_imagery_available:
            print(f"Using real imagery for simulation predictions in job {job['id']}")
            print("Note: Install 'deepforest' package for real AI tree detection")
        else:
            print(f"Running DeepForest in simulation mode for job {job['id']}")
        
        time.sleep(3)  # Simulate AI processing time
        
        # Generate realistic number of trees based on area
        area_acres = job['area_info']['acres']
        trees_per_acre = safe_random_uniform(80, 150)  # Typical range for tree density
        estimated_trees = int(area_acres * trees_per_acre)
        
        print(f"Generating {estimated_trees} simulated tree predictions for {area_acres:.2f} acres")
        
        # Try to get image dimensions for better pixel coordinate mapping
        image_width, image_height = 800, 600  # Default dimensions
        if 'processed_image' in job['files'] and os.path.exists(job['files']['processed_image']):
            try:
                if HAS_PIL:
                    with Image.open(job['files']['processed_image']) as img:
                        image_width, image_height = img.size
                        print(f"Using actual image dimensions: {image_width}x{image_height}")
            except Exception as e:
                print(f"Could not get image dimensions, using defaults: {e}")
        
        # Generate sample predictions
        job['results']['predictions'] = generate_sample_predictions(estimated_trees, job['geofence'], image_width, image_height)
        job['results']['raw_prediction_count'] = len(job['results']['predictions'])
        model_status = "REAL_IMAGERY_SIMULATION" if real_imagery_available else "SIMULATION"
        job['results']['model_used'] = f"{job['model']} ({model_status})"
        
        print(f"✓ Generated {len(job['results']['predictions'])} tree predictions")
        return
        
    # Legacy real DeepForest processing (fallback)
    print(f"Using legacy DeepForest model {job['model']} for job {job['id']}")
    
    try:
        from deepforest import main as deepforest_main
        from PIL import Image
        
        # Initialize DeepForest model
        model = deepforest_main.deepforest()
        
        # Try to load custom checkpoint if available
        model_filename = AVAILABLE_MODELS.get(job['model'], 'model_epoch_55.ckpt')
        checkpoint_path = find_model_path(model_filename)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading custom model: {checkpoint_path}")
            # Use the class method correctly
            model = deepforest_main.deepforest.load_from_checkpoint(checkpoint_path)
        else:
            print("Using default pre-trained model")
            model.use_release()
        
        # Load the processed image
        processed_image_path = job['files']['processed_image']
        
        if not os.path.exists(processed_image_path):
            raise FileNotFoundError(f"Processed image not found: {processed_image_path}")
        
        # Load image with PIL (DeepForest expects PIL or numpy array)
        image = Image.open(processed_image_path)
        print(f"Loaded processed image: {image.size} pixels")
        
        # Run AI predictions on the processed (normalized + masked) image
        predictions_df = model.predict_tile(
            processed_image_path,
            patch_size=job['settings'].get('patch_size', 1000),
            patch_overlap=job['settings'].get('patch_overlap', 0.35),
            iou_threshold=job['settings'].get('iou_threshold', 0.25)
        )
        
        print(f"   Raw predictions: {len(predictions_df)} detections")
        predictions = convert_deepforest_predictions(predictions_df, job)
        
        # Apply post-processing filters
        if job['settings'].get('enable_postprocessing', True):
            predictions_filtered = apply_post_processing_filters(predictions, job['settings'])
        else:
            predictions_filtered = predictions
            
        print(f"   Final predictions after post-processing: {len(predictions_filtered)} trees")
        
        # Store results
        job['results']['predictions'] = predictions_filtered
        job['results']['raw_prediction_count'] = len(predictions_df)
        job['results']['filtered_prediction_count'] = len(predictions_filtered)
        job['results']['model_used'] = f"{job['model']} (LEGACY_DEEPFOREST)"
        
    except Exception as e:
        print(f"Error in legacy DeepForest processing: {e}")
        print("Falling back to simulation mode...")
        
        # Fall back to simulation mode if AI processing fails
        area_acres = job['area_info']['acres']
        trees_per_acre = safe_random_uniform(80, 150)
        estimated_trees = int(area_acres * trees_per_acre)
        
        # Try to get image dimensions for better pixel coordinate mapping
        image_width, image_height = 800, 600  # Default dimensions
        if 'processed_image' in job['files'] and os.path.exists(job['files']['processed_image']):
            try:
                if HAS_PIL:
                    with Image.open(job['files']['processed_image']) as img:
                        image_width, image_height = img.size
            except:
                pass
        
        job['results']['predictions'] = generate_sample_predictions(estimated_trees, job['geofence'], image_width, image_height)
        job['results']['raw_prediction_count'] = len(job['results']['predictions'])
        job['results']['model_used'] = f"{job['model']} (FALLBACK_SIMULATION)"
        job['results']['fallback_reason'] = str(e)

def convert_deepforest_predictions(predictions_df, job):
    """Convert DeepForest predictions to our standard format"""
    if predictions_df.empty:
        return []
    
    predictions = []
    
    # Use normalized metadata if available (for proper coordinate conversion of processed images)
    normalized_metadata_path = None
    if 'normalized_metadata' in job['files']:
        normalized_metadata_path = job['files']['normalized_metadata']
    
    metadata_path = normalized_metadata_path if normalized_metadata_path and os.path.exists(normalized_metadata_path) else job['files']['metadata']
    
    # Load metadata to convert pixel coordinates to lat/lon
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"   Using metadata for coordinate conversion: {metadata_path}")
        if normalized_metadata_path:
            print(f"   ✓ Using normalized metadata for proper processed image coordinate mapping")
        
        bounds = metadata['bounds']
        image_size = metadata.get('image_size', metadata)
        width_px = image_size.get('width', 800)
        height_px = image_size.get('height', 600)
        
    except Exception as e:
        print(f"Warning: Could not load metadata for coordinate conversion: {e}")
        # Use geofence bounds as fallback
        geofence = job['geofence']
        bounds = {
            'min_lat': min(coord[1] for coord in geofence),
            'max_lat': max(coord[1] for coord in geofence),
            'min_lon': min(coord[0] for coord in geofence),
            'max_lon': max(coord[0] for coord in geofence)
        }
        width_px, height_px = 800, 600
    
    for _, row in predictions_df.iterrows():
        # Convert pixel coordinates to lat/lon
        center_x = (row['xmin'] + row['xmax']) / 2
        center_y = (row['ymin'] + row['ymax']) / 2
        
        # Convert to geographic coordinates
        lon = bounds['min_lon'] + (center_x / width_px) * (bounds['max_lon'] - bounds['min_lon'])
        lat = bounds['max_lat'] - (center_y / height_px) * (bounds['max_lat'] - bounds['min_lat'])
        
        # Calculate box dimensions
        box_width = row['xmax'] - row['xmin']
        box_height = row['ymax'] - row['ymin']
        
        prediction = {
            'latitude': lat,
            'longitude': lon,
            'confidence': float(row['score']),
            'bbox_pixels': {
                'xmin': int(row['xmin']),
                'ymin': int(row['ymin']),
                'xmax': int(row['xmax']),
                'ymax': int(row['ymax']),
                'width': int(box_width),
                'height': int(box_height)
            },
            'label': row.get('label', 'Tree'),
            'source': 'DeepForest_AI'
        }
        
        predictions.append(prediction)
    
    return predictions

def apply_post_processing_filters(predictions, settings):
    """Apply post-processing filters for tree predictions"""
    if not predictions:
        return predictions
    
    filtered = []
    confidence_threshold = settings.get('confidence_threshold', 0.25)
    
    for pred in predictions:
        # Apply confidence threshold
        if pred['confidence'] >= confidence_threshold:
            filtered.append(pred)
    
    print(f"Post-processing: {len(predictions)} → {len(filtered)} predictions (confidence ≥ {confidence_threshold})")
    return filtered

def generate_sample_predictions(count, geofence_coords, image_width=800, image_height=600):
    """Generate sample tree predictions for MVP testing with pixel coordinates"""
    import random
    
    if HAS_GEO:
        # Use accurate polygon containment check
        from shapely.geometry import Polygon, Point
        polygon = Polygon(geofence_coords)
        bounds = polygon.bounds
    else:
        # Use simple bounding box approach
        lons = [coord[0] for coord in geofence_coords]
        lats = [coord[1] for coord in geofence_coords]
        bounds = (min(lons), min(lats), max(lons), max(lats))
    
    predictions = []
    attempts = 0
    max_attempts = count * 10  # Prevent infinite loop
    
    while len(predictions) < count and attempts < max_attempts:
        # Random point within bounding box
        lon = random.uniform(bounds[0], bounds[2])
        lat = random.uniform(bounds[1], bounds[3])
        
        # Check if point is within polygon
        point_in_polygon = True  # Default to true for simple bounding box
        
        if HAS_GEO:
            point = Point(lon, lat)
            point_in_polygon = polygon.contains(point)
        else:
            # Simple point-in-polygon test (ray casting algorithm)
            point_in_polygon = point_in_polygon_simple(lon, lat, geofence_coords)
        
        if point_in_polygon:
            # Map lat/lon to approximate pixel coordinates
            # This is a simple mapping for visualization - real implementation would use image metadata
            center_x = int((lon - bounds[0]) / (bounds[2] - bounds[0]) * image_width)
            center_y = int((bounds[3] - lat) / (bounds[3] - bounds[1]) * image_height)  # Flip Y
            
            # Generate realistic bounding box size (typical tree crown: 10-30 pixels at 0.075 MPP)
            bbox_size = random.randint(10, 30)
            
            predictions.append({
                'longitude': lon,
                'latitude': lat,
                'confidence': random.uniform(0.3, 0.95),
                'bbox': {
                    'xmin': lon - 0.0001,
                    'ymin': lat - 0.0001,
                    'xmax': lon + 0.0001,
                    'ymax': lat + 0.0001
                },
                'bbox_pixels': {
                    'xmin': center_x - bbox_size // 2,
                    'ymin': center_y - bbox_size // 2,
                    'xmax': center_x + bbox_size // 2,
                    'ymax': center_y + bbox_size // 2,
                    'width': bbox_size,
                    'height': bbox_size
                },
                'source': 'Simulation'
            })
        attempts += 1
    
    return predictions

def point_in_polygon_simple(x, y, polygon_coords):
    """Simple point-in-polygon test using ray casting algorithm"""
    n = len(polygon_coords)
    inside = False
    
    p1x, p1y = polygon_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def generate_outputs_step(job):
    """Step 4: Generate CSV and visualization files"""
    import time
    time.sleep(1)
    
    print(f"Generating output files for job {job['id']}")
    
    predictions = job['results'].get('predictions', [])
    
    # Apply post-processing if enabled
    if job['settings'].get('enable_postprocessing', True):
        print("Applying post-processing filters...")
        # Apply confidence filtering
        conf_threshold = job['settings'].get('confidence_threshold', 0.25)
        before_count = len(predictions)
        predictions = [p for p in predictions if p['confidence'] >= conf_threshold]
        print(f"Confidence filter: {before_count} → {len(predictions)} predictions")
        
        # Apply additional post-processing logic
        predictions = apply_postprocessing(predictions, job['settings'])
        print(f"Overlap filter: → {len(predictions)} final predictions")
    
    job['results']['final_predictions'] = predictions
    job['results']['tree_count'] = len(predictions)
    job['results']['avg_confidence'] = safe_mean([p['confidence'] for p in predictions]) if predictions else 0
    
    # Generate actual output files
    job_temp_dir = os.path.join('temp_files', job['id'])
    
    # Generate CSV file
    csv_path = os.path.join(job_temp_dir, 'tree_predictions.csv')
    generate_csv_file(job, csv_path)
    
    # Generate KML file
    kml_path = os.path.join(job_temp_dir, 'geofence_boundary.kml')
    generate_kml_file(job, kml_path)
    
    # Generate summary JSON
    summary_path = os.path.join(job_temp_dir, 'processing_summary.json')
    generate_summary_file(job, summary_path)
    
    # Generate visualization image with tree predictions and bounding boxes
    viz_path = os.path.join(job_temp_dir, 'tree_detection_visualization.jpg')
    processed_img_path = job['files'].get('processed_image')  # Use processed image as background if available
    create_visualization_image(predictions, job['geofence'], viz_path, job['area_info']['acres'], processed_img_path)
    
    # Update file paths - collect all files generated during the process
    job['files'].update({
        'predictions_csv': csv_path,
        'geofence_kml': kml_path,
        'results_summary': summary_path,
        'visualization_image': viz_path
    })
    
    # Remove any None or non-existent files from the files list
    files_to_remove = []
    for file_type, file_path in job['files'].items():
        if not file_path or not os.path.exists(file_path):
            files_to_remove.append(file_type)
    
    for file_type in files_to_remove:
        job['files'].pop(file_type)
    
    print(f"✓ Generated {len(job['files'])} output files")
    print(f"Available files: {list(job['files'].keys())}")

def apply_postprocessing(predictions, settings):
    """Apply post-processing filters for tree predictions"""
    if not predictions:
        return predictions
    
    # For MVP, just apply simple confidence and overlap filtering
    # In full implementation, this would include:
    # - IOU-based duplicate removal
    # - Containment filtering
    # - Spatial clustering
    
    iou_threshold = settings.get('iou_threshold', 0.7)
    containment_threshold = settings.get('containment_threshold', 0.75)
    
    # Simple overlap removal for MVP
    filtered_predictions = []
    for pred in predictions:
        # Check for overlaps with existing predictions
        overlap_found = False
        for existing in filtered_predictions:
            # Use pixel-based overlap check if bbox_pixels are available
            if 'bbox_pixels' in pred and 'bbox_pixels' in existing:
                # Calculate overlap using bounding boxes
                bbox1 = pred['bbox_pixels']
                bbox2 = existing['bbox_pixels']
                
                # Calculate intersection
                x1 = max(bbox1['xmin'], bbox2['xmin'])
                y1 = max(bbox1['ymin'], bbox2['ymin'])
                x2 = min(bbox1['xmax'], bbox2['xmax'])
                y2 = min(bbox1['ymax'], bbox2['ymax'])
                
                if x1 < x2 and y1 < y2:  # Boxes overlap
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (bbox1['xmax'] - bbox1['xmin']) * (bbox1['ymax'] - bbox1['ymin'])
                    area2 = (bbox2['xmax'] - bbox2['xmin']) * (bbox2['ymax'] - bbox2['ymin'])
                    union = area1 + area2 - intersection
                    
                    if union > 0:
                        iou = intersection / union
                        if iou > iou_threshold:  # Significant overlap
                            # Keep the higher confidence prediction
                            if pred['confidence'] > existing['confidence']:
                                filtered_predictions.remove(existing)
                                break
                            else:
                                overlap_found = True
                                break
            else:
                # Fall back to distance-based overlap check
                dist = ((pred['longitude'] - existing['longitude'])**2 + 
                       (pred['latitude'] - existing['latitude'])**2)**0.5
                if dist < 0.00001:  # Much smaller threshold - only very close duplicates
                    # Keep the higher confidence prediction
                    if pred['confidence'] > existing['confidence']:
                        filtered_predictions.remove(existing)
                        break
                    else:
                        overlap_found = True
                        break
        
        if not overlap_found:
            filtered_predictions.append(pred)
    
    return filtered_predictions

def generate_csv_output(job):
    """Generate CSV file with prediction results (legacy function)"""
    predictions = job['results'].get('final_predictions', [])
    
    # Create CSV data
    csv_data = []
    for i, pred in enumerate(predictions):
        csv_data.append({
            'tree_id': i + 1,
            'longitude': pred['longitude'],
            'latitude': pred['latitude'],
            'confidence': pred['confidence'],
            'bbox_xmin': pred['bbox']['xmin'],
            'bbox_ymin': pred['bbox']['ymin'],
            'bbox_xmax': pred['bbox']['xmax'],
            'bbox_ymax': pred['bbox']['ymax']
        })
    
    job['results']['csv_data'] = csv_data

def generate_csv_file(job, output_path):
    """Generate actual CSV file with tree predictions"""
    predictions = job['results'].get('final_predictions', [])
    
    # Create CSV data
    csv_data = []
    for i, pred in enumerate(predictions):
        # Handle both bbox_pixels and bbox format for compatibility
        bbox = pred.get('bbox_pixels', pred.get('bbox', {}))
        
        csv_data.append({
            'tree_id': i + 1,
            'longitude': pred['longitude'],
            'latitude': pred['latitude'],
            'confidence': round(pred['confidence'], 4),
            'bbox_xmin': bbox.get('xmin', 0),
            'bbox_ymin': bbox.get('ymin', 0),
            'bbox_xmax': bbox.get('xmax', 0),
            'bbox_ymax': bbox.get('ymax', 0),
            'area_acres': job['area_info']['acres'],
            'processing_date': job['created_at']
        })
    
    # Write to CSV file
    if csv_data:
        if HAS_PANDAS:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
        else:
            # Fallback CSV writing without pandas
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
        print(f"✓ Created CSV file with {len(csv_data)} predictions: {output_path}")
    else:
        # Create empty CSV with headers
        headers = ['tree_id', 'longitude', 'latitude', 'confidence', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax', 'area_acres', 'processing_date']
        if HAS_PANDAS:
            pd.DataFrame(columns=headers).to_csv(output_path, index=False)
        else:
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
        print(f"✓ Created empty CSV file: {output_path}")

def generate_kml_file(job, output_path):
    """Generate KML file with geofence boundary"""
    geofence_coords = job['geofence']
    
    # Simple KML structure
    kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>ArborNote Geofence - {job['id']}</name>
    <description>
      Processing Area: {job['area_info']['acres']} acres
      Cost: ${job['area_info']['cost_usd']}
      Trees Detected: {job['results'].get('tree_count', 0)}
      Processing Date: {job['created_at']}
    </description>
    <Placemark>
      <name>Processing Boundary</name>
      <description>Geofence area for tree detection</description>
      <Style>
        <LineStyle>
          <color>ff00ff00</color>
          <width>3</width>
        </LineStyle>
        <PolyStyle>
          <color>3300ff00</color>
        </PolyStyle>
      </Style>
      <Polygon>
        <extrude>1</extrude>
        <altitudeMode>relativeToGround</altitudeMode>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
'''
    
    # Add coordinates
    for lon, lat in geofence_coords:
        kml_content += f"              {lon},{lat},0\n"
    
    # Close first coordinate to complete polygon
    if geofence_coords:
        first_lon, first_lat = geofence_coords[0]
        kml_content += f"              {first_lon},{first_lat},0\n"
    
    kml_content += '''            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>'''
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(kml_content)
    
    print(f"✓ Created KML file: {output_path}")

def create_kml_from_geofence(geofence_coords, kml_path):
    """Create a KML file from geofence coordinates"""
    # Ensure we have at least 3 coordinates and close the polygon
    coords = list(geofence_coords)
    if len(coords) < 3:
        raise ValueError(f"KML polygon must contain at least 3 coordinates, got {len(coords)}")
    
    # Ensure polygon is closed (first and last point should be the same)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    # Format coordinates as lon,lat,altitude (altitude=0)
    coord_strings = []
    for coord in coords:
        if len(coord) >= 2:
            lon, lat = coord[0], coord[1]
            coord_strings.append(f"{lon},{lat},0")
    
    coordinates_text = " ".join(coord_strings)
    
    kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>Geofence Boundary</name>
        <Placemark>
            <name>Processing Area</name>
            <Polygon>
                <outerBoundaryIs>
                    <LinearRing>
                        <coordinates>{coordinates_text}</coordinates>
                    </LinearRing>
                </outerBoundaryIs>
            </Polygon>
        </Placemark>
    </Document>
</kml>'''
    
    with open(kml_path, 'w') as f:
        f.write(kml_content)
    
    print(f"✓ Created KML file with {len(coords)} coordinates: {kml_path}")
    
    # Debug: show first few coordinates
    print(f"   Sample coordinates: {coordinates_text[:100]}...")

def generate_summary_file(job, output_path):
    """Generate JSON summary file with all processing details"""
    summary = {
        'job_id': job['id'],
        'processing_info': {
            'created_at': job['created_at'],
            'completed_at': datetime.now().isoformat(),
            'status': job['status'],
            'model_used': job['results'].get('model_used', job['model']),
            'settings_used': job['settings']
        },
        'area_info': job['area_info'],
        'geofence': {
            'coordinates': job['geofence'],
            'coordinate_count': len(job['geofence'])
        },
        'results': {
            'tree_count': job['results'].get('tree_count', 0),
            'raw_predictions': job['results'].get('raw_prediction_count', 0),
            'avg_confidence': round(job['results'].get('avg_confidence', 0), 4),
            'predictions_filtered': job['settings'].get('enable_postprocessing', False)
        },
        'files_generated': list(job['files'].keys()),
        'simulation_mode': not (HAS_NEARMAP and HAS_DEEPFOREST and HAS_BLACKOUT) or FORCE_SIMULATION_MODE
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Created summary file: {output_path}")

def resample_image_to_target_resolution(input_path, output_path, scale_factor, target_mpp=0.075):
    """Resample image to target resolution for consistent AI predictions"""
    if not HAS_PIL:
        # Fall back to just copying the file
        shutil.copy2(input_path, output_path)
        print(f"Warning: PIL not available, copying image without resampling")
        return
    
    try:
        with Image.open(input_path) as img:
            original_size = img.size
            
            # Calculate new size based on scale factor
            new_width = int(original_size[0] * scale_factor)
            new_height = int(original_size[1] * scale_factor)
            
            print(f"Resampling image from {original_size[0]}x{original_size[1]} to {new_width}x{new_height}")
            print(f"Scale factor: {scale_factor:.4f} (target: {target_mpp} m/px)")
            
            # Use high-quality resampling for better results
            # LANCZOS provides good quality for both upsampling and downsampling
            resampled_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save with high quality
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                resampled_img.save(output_path, 'JPEG', quality=95, optimize=True)
            else:
                resampled_img.save(output_path)
                
            print(f"✓ Successfully resampled image to target resolution: {output_path}")
            
    except Exception as e:
        print(f"Error resampling image: {e}")
        print("Falling back to copying original image...")
        shutil.copy2(input_path, output_path)

def create_normalized_metadata(original_metadata_path, output_metadata_path, scale_factor, target_mpp):
    """Create updated metadata for normalized image with corrected dimensions and coordinates"""
    try:
        # Load original metadata
        with open(original_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update resolution
        metadata['resolution_mpp'] = target_mpp
        metadata['target_resolution_meters_per_pixel'] = target_mpp
        metadata['normalized'] = True
        metadata['scale_factor_applied'] = scale_factor
        metadata['original_metadata_path'] = original_metadata_path
        
        # Update image dimensions if available
        if 'image_size' in metadata:
            original_width = metadata['image_size']['width']
            original_height = metadata['image_size']['height']
            
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            metadata['image_size']['width'] = new_width
            metadata['image_size']['height'] = new_height
            metadata['original_image_size'] = {'width': original_width, 'height': original_height}
            
            print(f"   📐 Updated image dimensions: {original_width}x{original_height} → {new_width}x{new_height}")
        
        # Save normalized metadata
        with open(output_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"   ✅ Created normalized metadata: {output_metadata_path}")
        return metadata
        
    except Exception as e:
        print(f"   ❌ Error creating normalized metadata: {e}")
        # Fall back to copying original metadata
        import shutil
        shutil.copy2(original_metadata_path, output_metadata_path)
        return None
    

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('temp_files', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Production mode detection
    is_production = os.getenv('FLASK_ENV') == 'production'
    
    print("=" * 60)
    print("🌳 ArborNote Tree Detection System v3.0")
    print("=" * 60)
    print(f"🔧 Environment: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
    print(f"✅ Flask: Available")
    print(f"{'✅' if HAS_SOCKETIO else '⚠️ '} SocketIO: {'Available' if HAS_SOCKETIO else 'Not available (polling mode)'}")
    print(f"{'✅' if HAS_GEO else '⚠️ '} Geospatial: {'Available' if HAS_GEO else 'Approximate calculations'}")
    print(f"{'✅' if HAS_PIL else '⚠️ '} PIL/Imaging: {'Available' if HAS_PIL else 'Text placeholders only'}")
    
    # Check for production modules
    try:
        from label_process.extract_nearmap import NearmapTileExtractor
        HAS_NEARMAP = True
        nearmap_status = "✅ Available"
    except ImportError:
        HAS_NEARMAP = False
        nearmap_status = "⚠️  Simulation mode"
    
    try:
        import deepforest
        HAS_DEEPFOREST = True
        deepforest_status = "✅ Available"
    except ImportError:
        HAS_DEEPFOREST = False
        deepforest_status = "⚠️  Simulation mode"
    
    print(f"{nearmap_status[:2]} Nearmap: {nearmap_status[3:]}")
    print(f"{deepforest_status[:2]} DeepForest: {deepforest_status[3:]}")
    
    # Check Production Pipeline
    try:
        from label_process.production_pipeline import ProductionPipeline
        HAS_PRODUCTION_PIPELINE = True
        pipeline_status = "✅ Production Pipeline Available"
    except ImportError:
        HAS_PRODUCTION_PIPELINE = False
        pipeline_status = "⚠️  Using Legacy Processing"
    
    print(f"{pipeline_status[:2]} Processing: {pipeline_status[3:]}")
    
    # Check API key
    api_key = os.getenv('NEARMAP_API_KEY')
    if api_key:
        print(f"🔑 Nearmap API: Configured ({api_key[:10]}...)")
    else:
        print("⚠️  Nearmap API: Not configured (will use simulation mode)")
    
    # Check model files
    model_found = False
    for model_name, filename in AVAILABLE_MODELS.items():
        model_path = find_model_path(filename)
        if model_path:
            model_found = True
            print(f"🤖 Model {model_name}: ✅ {model_path}")
            break
    
    if not model_found:
        print("⚠️  No model checkpoints found - using simulation mode")
        print("   Expected locations:")
        print("     - ../checkpoints/")
        print("     - ./checkpoints/")
    
    print("=" * 60)
    
    # Start server based on environment
    if is_production:
        print("🏭 Production mode: Optimized Flask server for Windows")
        print("   Debug disabled, security hardened")
        app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'arbornote-production-secret-key-2025-secure-random-string')
        print()
    else:
        print("🔧 Development mode: Debug enabled")
        print()
    
    # Configure and start the app
    debug_mode = not is_production
    host = os.getenv('FLASK_HOST', config.get('app', {}).get('host', '0.0.0.0'))
    port = int(os.getenv('FLASK_PORT', config.get('app', {}).get('port', 4000)))
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"🌐 Access at: http://localhost:{port}")
    
    if HAS_SOCKETIO and socketio:
        socketio.run(app, debug=debug_mode, host=host, port=port, allow_unsafe_werkzeug=True)
    else:
        print("⚠️  Running without WebSocket support (using polling for updates)")
        app.run(debug=debug_mode, host=host, port=port)
