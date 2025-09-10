#!/usr/bin/env python3
"""
Production Palm Detection Pipeline
Class-based implementation for use with Flask app and standalone processing
"""

import os
import cv2
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Optional imports with graceful fallbacks
try:
    from deepforest import main as deepforest_main
    HAS_DEEPFOREST = True
except ImportError:
    print("Warning: DeepForest not available")
    HAS_DEEPFOREST = False

try:
    from shapely.geometry import box, Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    print("Warning: Shapely not available")
    HAS_SHAPELY = False

try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    print("Warning: XML parser not available")
    HAS_XML = False


@dataclass
class ProcessingConfig:
    """Configuration for the palm detection pipeline"""
    # Model settings
    model_path: str = ""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.9
    patch_size: int = 1000
    patch_overlap: float = 0.15
    
    # Post-processing settings
    enable_postprocessing: bool = True
    containment_threshold: float = 0.75
    enable_adaptive_filtering: bool = True
    adaptive_k_factor: float = 2.0
    
    # Geofencing settings
    enable_geofencing: bool = True
    
    # Visualization settings
    show_removed_boxes: bool = True
    save_intermediate_results: bool = True
    
    # Output settings
    output_format: str = "jpg"
    save_csv: bool = True
    save_summary: bool = True


@dataclass
class ProcessingResults:
    """Results from the palm detection pipeline"""
    # Prediction counts
    initial_predictions: int = 0
    confidence_filtered: int = 0
    geofence_filtered: int = 0
    adaptive_filtered: int = 0
    postprocessed: int = 0
    final_predictions: int = 0
    
    # Removed predictions by category
    geofence_removed: List = None
    adaptive_removed: List = None
    postprocessing_removed: List = None
    
    # Processing metadata
    adaptive_threshold: float = 0.25
    processing_time: float = 0.0
    model_used: str = ""
    
    def __post_init__(self):
        if self.geofence_removed is None:
            self.geofence_removed = []
        if self.adaptive_removed is None:
            self.adaptive_removed = []
        if self.postprocessing_removed is None:
            self.postprocessing_removed = []


class ProductionPipeline:
    """
    Production-ready palm detection pipeline with configurable parameters
    Integrates with Flask app and supports standalone usage
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the pipeline with configuration"""
        self.config = config or ProcessingConfig()
        
        # State management
        self.model = None
        self.geofence_polygon = None
        self.image_bounds = None
        self.image_width = 0
        self.image_height = 0
        
        # Results tracking
        self.results = ProcessingResults()
        self.predictions_raw = []
        self.predictions_final = []
        
        # Processing history for debugging
        self.processing_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Add a message to the processing log"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.processing_log.append(log_entry)
        print(log_entry)
    
    def load_model(self, model_path: str) -> bool:
        """Load DeepForest model from checkpoint"""
        if not HAS_DEEPFOREST:
            self.log("DeepForest not available, using simulation mode", "WARNING")
            return False
            
        try:
            self.log(f"Loading DeepForest model from: {model_path}")
            self.model = deepforest_main.deepforest.load_from_checkpoint(model_path)
            
            # Configure model parameters
            self.model.config["patch_size"] = self.config.patch_size
            self.model.config["patch_overlap"] = self.config.patch_overlap
            
            self.results.model_used = os.path.basename(model_path)
            self.log(f"Model loaded successfully: patch_size={self.config.patch_size}, overlap={self.config.patch_overlap}")
            return True
            
        except Exception as e:
            self.log(f"Error loading model: {e}", "ERROR")
            return False
    
    def load_geofence_from_kml(self, kml_path: str) -> bool:
        """Load geofence polygon from KML file"""
        if not HAS_XML or not HAS_SHAPELY:
            self.log("XML parser or Shapely not available for geofencing", "WARNING")
            return False
            
        if not os.path.exists(kml_path):
            self.log(f"KML file not found: {kml_path}", "WARNING")
            return False
            
        try:
            # Parse KML file
            tree = ET.parse(kml_path)
            root = tree.getroot()
            
            # Handle KML namespace
            namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            # Look for coordinates
            coordinates_text = None
            polygon_coords = root.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespace)
            if polygon_coords is not None:
                coordinates_text = polygon_coords.text
            else:
                # Try without namespace
                polygon_coords = root.find('.//coordinates')
                if polygon_coords is not None:
                    coordinates_text = polygon_coords.text
            
            if coordinates_text is None:
                self.log(f"No coordinates found in KML file: {kml_path}", "ERROR")
                return False
            
            # Parse coordinates
            coord_points = []
            for coord_str in coordinates_text.strip().split():
                coord_parts = coord_str.split(',')
                if len(coord_parts) >= 2:
                    try:
                        lon = float(coord_parts[0])
                        lat = float(coord_parts[1])
                        coord_points.append((lon, lat))
                    except ValueError:
                        continue
            
            if len(coord_points) < 3:
                self.log(f"Insufficient coordinates in KML (need ≥3, found {len(coord_points)})", "ERROR")
                return False
            
            # Create Shapely polygon
            self.geofence_polygon = Polygon(coord_points)
            self.log(f"Loaded geofence with {len(coord_points)} points, bounds: {self.geofence_polygon.bounds}")
            return True
            
        except Exception as e:
            self.log(f"Error parsing KML file: {e}", "ERROR")
            return False
    
    def load_image_metadata(self, metadata_path: str) -> bool:
        """Load image metadata for coordinate conversion"""
        if not os.path.exists(metadata_path):
            self.log(f"Metadata file not found: {metadata_path}", "WARNING")
            return False
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Handle different metadata formats
            if 'bounds' in metadata:
                bounds = metadata['bounds']
                self.image_bounds = {
                    'min_lon': bounds['min_lon'],
                    'max_lon': bounds['max_lon'],
                    'min_lat': bounds['min_lat'],
                    'max_lat': bounds['max_lat']
                }
            elif 'min_lon' in metadata:
                self.image_bounds = {
                    'min_lon': metadata['min_lon'],
                    'max_lon': metadata['max_lon'],
                    'min_lat': metadata['min_lat'],
                    'max_lat': metadata['max_lat']
                }
            else:
                self.log("Metadata format not recognized", "ERROR")
                return False
            
            self.log(f"Loaded image bounds: {self.image_bounds}")
            return True
            
        except Exception as e:
            self.log(f"Error loading metadata: {e}", "ERROR")
            return False
    
    def pixel_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to lat/lon"""
        if not self.image_bounds:
            return 0.0, 0.0
            
        lon_per_pixel = (self.image_bounds['max_lon'] - self.image_bounds['min_lon']) / self.image_width
        lat_per_pixel = (self.image_bounds['max_lat'] - self.image_bounds['min_lat']) / self.image_height
        
        lon = self.image_bounds['min_lon'] + (x * lon_per_pixel)
        lat = self.image_bounds['max_lat'] - (y * lat_per_pixel)
        
        return lat, lon
    
    def run_predictions(self, image_path: str) -> bool:
        """Run DeepForest predictions on image"""
        if not self.model:
            self.log("No model loaded, using simulation mode", "WARNING")
            return self._generate_simulation_predictions(image_path)
        
        try:
            self.log(f"Running predictions on: {image_path}")
            self.log(f"Using patch_size={self.config.patch_size}, overlap={self.config.patch_overlap}, iou={self.config.iou_threshold}")
            
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                self.log(f"Could not load image: {image_path}", "ERROR")
                return False
            
            self.image_height, self.image_width = image.shape[:2]
            self.log(f"Image dimensions: {self.image_width}x{self.image_height}")
            
            # Run predictions
            pred_df = self.model.predict_tile(
                image_path,
                patch_size=self.config.patch_size,
                patch_overlap=self.config.patch_overlap,
                iou_threshold=self.config.iou_threshold
            )
            
            # Handle empty results
            if pred_df is None or len(pred_df) == 0:
                self.log("No predictions returned by DeepForest", "WARNING")
                pred_df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label'])
            
            self.results.initial_predictions = len(pred_df)
            
            # Apply confidence threshold
            pred_df = pred_df[pred_df["score"] > self.config.confidence_threshold]
            self.results.confidence_filtered = len(pred_df)
            
            # Convert to internal format
            self.predictions_raw = []
            for _, row in pred_df.iterrows():
                if HAS_SHAPELY:
                    pred_box = box(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                    center = Point((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2)
                else:
                    pred_box = {
                        'xmin': row['xmin'], 'ymin': row['ymin'],
                        'xmax': row['xmax'], 'ymax': row['ymax']
                    }
                    center = {
                        'x': (row['xmin'] + row['xmax']) / 2,
                        'y': (row['ymin'] + row['ymax']) / 2
                    }
                
                self.predictions_raw.append({
                    'box': pred_box,
                    'center': center,
                    'confidence': row['score'],
                    'raw_data': row
                })
            
            self.log(f"Predictions: {self.results.initial_predictions} initial → {self.results.confidence_filtered} after confidence filter")
            return True
            
        except Exception as e:
            self.log(f"Error running predictions: {e}", "ERROR")
            return False
    
    def _generate_simulation_predictions(self, image_path: str) -> bool:
        """Generate simulation predictions for testing"""
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                self.image_width, self.image_height = 1000, 800
            else:
                self.image_height, self.image_width = image.shape[:2]
            
            # Generate random predictions
            import random
            num_predictions = random.randint(10, 50)
            
            self.predictions_raw = []
            for i in range(num_predictions):
                x1 = random.randint(0, self.image_width - 100)
                y1 = random.randint(0, self.image_height - 100)
                x2 = x1 + random.randint(20, 80)
                y2 = y1 + random.randint(20, 80)
                confidence = random.uniform(0.1, 0.9)
                
                if HAS_SHAPELY:
                    pred_box = box(x1, y1, x2, y2)
                    center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                else:
                    pred_box = {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
                    center = {'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2}
                
                self.predictions_raw.append({
                    'box': pred_box,
                    'center': center,
                    'confidence': confidence,
                    'raw_data': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2, 'score': confidence}
                })
            
            self.results.initial_predictions = num_predictions
            self.results.confidence_filtered = len([p for p in self.predictions_raw if p['confidence'] > self.config.confidence_threshold])
            
            self.log(f"Generated {num_predictions} simulation predictions")
            return True
            
        except Exception as e:
            self.log(f"Error generating simulation: {e}", "ERROR")
            return False
    
    def apply_geofence_filtering(self) -> bool:
        """Filter predictions by geofence polygon"""
        if not self.config.enable_geofencing or not self.geofence_polygon or not self.image_bounds:
            self.log("Geofence filtering skipped (disabled or no geofence/metadata)")
            self.results.geofence_filtered = len(self.predictions_raw)
            return True
        
        try:
            self.log("Applying geofence filtering...")
            
            filtered_predictions = []
            removed_predictions = []
            
            for pred in self.predictions_raw:
                # Get center coordinates
                if HAS_SHAPELY:
                    center_x = pred['center'].x
                    center_y = pred['center'].y
                else:
                    center_x = pred['center']['x']
                    center_y = pred['center']['y']
                
                # Convert to lat/lon
                lat, lon = self.pixel_to_latlon(center_x, center_y)
                
                # Check if point is within geofence
                if HAS_SHAPELY:
                    geo_point = Point(lon, lat)
                    if self.geofence_polygon.contains(geo_point):
                        filtered_predictions.append(pred)
                    else:
                        removed_predictions.append(pred)
                else:
                    # Simple fallback without Shapely
                    filtered_predictions.append(pred)
            
            self.predictions_raw = filtered_predictions
            self.results.geofence_removed = removed_predictions
            self.results.geofence_filtered = len(filtered_predictions)
            
            removed_count = len(removed_predictions)
            self.log(f"Geofence filtering: removed {removed_count} predictions, {len(filtered_predictions)} remaining")
            return True
            
        except Exception as e:
            self.log(f"Error in geofence filtering: {e}", "ERROR")
            return False
    
    def apply_adaptive_filtering(self) -> bool:
        """Apply adaptive confidence filtering based on prediction statistics"""
        if not self.config.enable_adaptive_filtering or not self.predictions_raw:
            self.log("Adaptive filtering skipped (disabled or no predictions)")
            self.results.adaptive_filtered = len(self.predictions_raw)
            return True
        
        try:
            self.log("Applying adaptive confidence filtering...")
            
            # Calculate statistics
            confidences = [pred['confidence'] for pred in self.predictions_raw]
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            
            # Calculate adaptive threshold
            adaptive_threshold = max(0.25, mean_conf - self.config.adaptive_k_factor * std_conf)
            self.results.adaptive_threshold = adaptive_threshold
            
            self.log(f"Confidence stats: μ={mean_conf:.3f}, σ={std_conf:.3f}")
            self.log(f"Adaptive threshold: max(0.25, {mean_conf:.3f} - {self.config.adaptive_k_factor}×{std_conf:.3f}) = {adaptive_threshold:.3f}")
            
            # Apply filtering
            filtered_predictions = []
            removed_predictions = []
            
            for pred in self.predictions_raw:
                if pred['confidence'] >= adaptive_threshold:
                    filtered_predictions.append(pred)
                else:
                    removed_predictions.append(pred)
            
            self.predictions_raw = filtered_predictions
            self.results.adaptive_removed = removed_predictions
            self.results.adaptive_filtered = len(filtered_predictions)
            
            removed_count = len(removed_predictions)
            percentage_kept = (len(filtered_predictions) / (len(filtered_predictions) + removed_count)) * 100 if (len(filtered_predictions) + removed_count) > 0 else 0
            
            self.log(f"Adaptive filtering: removed {removed_count} predictions ({percentage_kept:.1f}% kept)")
            return True
            
        except Exception as e:
            self.log(f"Error in adaptive filtering: {e}", "ERROR")
            return False
    
    def apply_postprocessing(self) -> bool:
        """Apply post-processing to remove contained/overlapping boxes"""
        if not self.config.enable_postprocessing or not self.predictions_raw:
            self.log("Post-processing skipped (disabled or no predictions)")
            self.results.postprocessed = len(self.predictions_raw)
            return True
        
        try:
            self.log(f"Applying post-processing with containment threshold {self.config.containment_threshold}")
            
            # For now, implement a simple version that works with/without Shapely
            if HAS_SHAPELY:
                return self._apply_postprocessing_shapely()
            else:
                return self._apply_postprocessing_simple()
                
        except Exception as e:
            self.log(f"Error in post-processing: {e}", "ERROR")
            return False
    
    def _apply_postprocessing_shapely(self) -> bool:
        """Post-processing implementation using Shapely"""
        predictions = self.predictions_raw.copy()
        boxes_to_remove = set()
        
        for i, pred_i in enumerate(predictions):
            if i in boxes_to_remove:
                continue
                
            for j, pred_j in enumerate(predictions):
                if i == j or j in boxes_to_remove:
                    continue
                
                # Calculate intersection
                intersection_area = pred_i['box'].intersection(pred_j['box']).area
                area_i = pred_i['box'].area
                area_j = pred_j['box'].area
                
                if area_i > 0 and area_j > 0:
                    containment_i_in_j = intersection_area / area_i
                    containment_j_in_i = intersection_area / area_j
                    
                    # Remove contained boxes
                    if containment_i_in_j > self.config.containment_threshold:
                        boxes_to_remove.add(i)
                        break
                    elif containment_j_in_i > self.config.containment_threshold:
                        boxes_to_remove.add(j)
        
        # Create filtered lists
        filtered_predictions = []
        removed_predictions = []
        
        for i, pred in enumerate(predictions):
            if i in boxes_to_remove:
                removed_predictions.append(pred)
            else:
                filtered_predictions.append(pred)
        
        self.predictions_raw = filtered_predictions
        self.results.postprocessing_removed = removed_predictions
        self.results.postprocessed = len(filtered_predictions)
        
        self.log(f"Post-processing: removed {len(removed_predictions)} contained boxes")
        return True
    
    def _apply_postprocessing_simple(self) -> bool:
        """Simple post-processing without Shapely"""
        # For now, just pass through without modification
        self.results.postprocessed = len(self.predictions_raw)
        self.log("Post-processing: using simple mode (Shapely not available)")
        return True
    
    def create_visualization(self, image_path: str, output_path: str) -> bool:
        """Create visualization with bounding boxes"""
        try:
            self.log(f"Creating visualization: {output_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.log(f"Could not load image for visualization: {image_path}", "ERROR")
                return False
            
            # Draw final predictions in green
            for pred in self.predictions_raw:
                if HAS_SHAPELY:
                    x1, y1, x2, y2 = map(int, pred['box'].bounds)
                else:
                    x1, y1, x2, y2 = int(pred['box']['xmin']), int(pred['box']['ymin']), int(pred['box']['xmax']), int(pred['box']['ymax'])
                
                # Green box for final predictions
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence text
                conf_text = f"{pred['confidence']:.2f}"
                text_y = max(y1 - 10, 15)
                cv2.putText(image, conf_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw removed boxes if enabled
            if self.config.show_removed_boxes:
                self._draw_removed_boxes(image)
            
            # Save visualization
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            
            self.log(f"Visualization saved with {len(self.predictions_raw)} final predictions")
            return True
            
        except Exception as e:
            self.log(f"Error creating visualization: {e}", "ERROR")
            return False
    
    def _draw_removed_boxes(self, image):
        """Draw removed boxes on visualization"""
        # Draw geofence removed boxes in orange
        for pred in self.results.geofence_removed:
            if HAS_SHAPELY:
                x1, y1, x2, y2 = map(int, pred['box'].bounds)
            else:
                x1, y1, x2, y2 = int(pred['box']['xmin']), int(pred['box']['ymin']), int(pred['box']['xmax']), int(pred['box']['ymax'])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 1)  # Orange
            cv2.putText(image, f"G:{pred['confidence']:.2f}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 200), 1)
        
        # Draw adaptive removed boxes in purple
        for pred in self.results.adaptive_removed:
            if HAS_SHAPELY:
                x1, y1, x2, y2 = map(int, pred['box'].bounds)
            else:
                x1, y1, x2, y2 = int(pred['box']['xmin']), int(pred['box']['ymin']), int(pred['box']['xmax']), int(pred['box']['ymax'])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (128, 0, 128), 1)  # Purple
            cv2.putText(image, f"A:{pred['confidence']:.2f}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (64, 0, 64), 1)
        
        # Draw post-processing removed boxes in gray
        for pred in self.results.postprocessing_removed:
            if HAS_SHAPELY:
                x1, y1, x2, y2 = map(int, pred['box'].bounds)
            else:
                x1, y1, x2, y2 = int(pred['box']['xmin']), int(pred['box']['ymin']), int(pred['box']['xmax']), int(pred['box']['ymax'])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 1)  # Gray
            cv2.putText(image, f"R:{pred['confidence']:.2f}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (64, 64, 64), 1)
    
    def save_results(self, output_dir: str) -> Dict[str, str]:
        """Save all results and return file paths"""
        output_files = {}
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CSV file
            if self.config.save_csv:
                csv_path = os.path.join(output_dir, 'predictions.csv')
                self._save_csv(csv_path)
                output_files['csv'] = csv_path
            
            # Save summary JSON
            if self.config.save_summary:
                summary_path = os.path.join(output_dir, 'processing_summary.json')
                self._save_summary(summary_path)
                output_files['summary'] = summary_path
            
            return output_files
            
        except Exception as e:
            self.log(f"Error saving results: {e}", "ERROR")
            return {}
    
    def _save_csv(self, csv_path: str):
        """Save predictions to CSV file"""
        if not self.predictions_raw:
            # Create empty CSV with headers
            pd.DataFrame(columns=['id', 'longitude', 'latitude', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']).to_csv(csv_path, index=False)
            return
        
        prediction_data = []
        for i, pred in enumerate(self.predictions_raw):
            if HAS_SHAPELY:
                center_x = pred['center'].x
                center_y = pred['center'].y
                bounds = pred['box'].bounds
            else:
                center_x = pred['center']['x']
                center_y = pred['center']['y']
                bounds = [pred['box']['xmin'], pred['box']['ymin'], pred['box']['xmax'], pred['box']['ymax']]
            
            # Convert to lat/lon if possible
            lat, lon = self.pixel_to_latlon(center_x, center_y)
            
            prediction_data.append({
                'id': i + 1,
                'longitude': lon,
                'latitude': lat,
                'confidence': pred['confidence'],
                'xmin': bounds[0],
                'ymin': bounds[1],
                'xmax': bounds[2],
                'ymax': bounds[3]
            })
        
        df = pd.DataFrame(prediction_data)
        df.to_csv(csv_path, index=False)
        self.log(f"Saved {len(prediction_data)} predictions to CSV")
    
    def _save_summary(self, summary_path: str):
        """Save processing summary to JSON file"""
        summary = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.results.model_used,
                'config': {
                    'confidence_threshold': self.config.confidence_threshold,
                    'iou_threshold': self.config.iou_threshold,
                    'patch_size': self.config.patch_size,
                    'patch_overlap': self.config.patch_overlap,
                    'containment_threshold': self.config.containment_threshold,
                    'adaptive_threshold': self.results.adaptive_threshold
                }
            },
            'results': {
                'initial_predictions': self.results.initial_predictions,
                'confidence_filtered': self.results.confidence_filtered,
                'geofence_filtered': self.results.geofence_filtered,
                'adaptive_filtered': self.results.adaptive_filtered,
                'postprocessed': self.results.postprocessed,
                'final_predictions': len(self.predictions_raw)
            },
            'removed_counts': {
                'geofence_removed': len(self.results.geofence_removed),
                'adaptive_removed': len(self.results.adaptive_removed),
                'postprocessing_removed': len(self.results.postprocessing_removed)
            },
            'processing_log': self.processing_log
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"Saved processing summary to {summary_path}")
    
    def run_complete_pipeline(self, image_path: str, model_path: str, output_dir: str, 
                             kml_path: Optional[str] = None, metadata_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete processing pipeline
        
        Returns:
            Dictionary with results and file paths
        """
        start_time = datetime.now()
        
        self.log("=" * 60)
        self.log("STARTING PRODUCTION PALM DETECTION PIPELINE")
        self.log("=" * 60)
        
        # Update configuration
        self.config.model_path = model_path
        
        # Initialize results
        success = True
        output_files = {}
        
        try:
            # Step 1: Load model
            if not self.load_model(model_path):
                success = False
            
            # Step 2: Load geofence (optional)
            if kml_path:
                self.load_geofence_from_kml(kml_path)
            
            # Step 3: Load metadata (optional)
            if metadata_path:
                self.load_image_metadata(metadata_path)
            
            # Step 4: Run predictions
            if not self.run_predictions(image_path):
                success = False
            
            # Step 5: Apply filtering pipeline
            if not self.apply_geofence_filtering():
                success = False
            
            if not self.apply_adaptive_filtering():
                success = False
            
            if not self.apply_postprocessing():
                success = False
            
            # Step 6: Create visualization
            viz_path = os.path.join(output_dir, f'detection_visualization.{self.config.output_format}')
            if self.create_visualization(image_path, viz_path):
                output_files['visualization'] = viz_path
            
            # Step 7: Save results
            result_files = self.save_results(output_dir)
            output_files.update(result_files)
            
            # Update final results
            self.results.final_predictions = len(self.predictions_raw)
            self.results.processing_time = (datetime.now() - start_time).total_seconds()
            
            self.log("=" * 60)
            self.log("PIPELINE COMPLETE")
            self.log("=" * 60)
            self.log(f"Processing time: {self.results.processing_time:.2f} seconds")
            self.log(f"Final predictions: {self.results.final_predictions}")
            self.log(f"Files created: {list(output_files.keys())}")
            
            return {
                'success': success,
                'results': self.results,
                'predictions': self.predictions_raw,
                'output_files': output_files,
                'processing_log': self.processing_log
            }
            
        except Exception as e:
            self.log(f"Pipeline failed: {e}", "ERROR")
            return {
                'success': False,
                'error': str(e),
                'results': self.results,
                'output_files': output_files,
                'processing_log': self.processing_log
            }


def create_production_config(**kwargs) -> ProcessingConfig:
    """Helper function to create production configuration"""
    return ProcessingConfig(**kwargs)


# Example standalone usage
if __name__ == "__main__":
    # Example configuration
    config = create_production_config(
        confidence_threshold=0.25,
        iou_threshold=0.9,
        patch_size=1000,
        patch_overlap=0.15,
        enable_postprocessing=True,
        containment_threshold=0.75
    )
    
    # Create pipeline
    pipeline = ProductionPipeline(config)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        image_path="path/to/image.jpg",
        model_path="path/to/model.ckpt",
        output_dir="outputs/",
        kml_path="path/to/geofence.kml",  # Optional
        metadata_path="path/to/metadata.json"  # Optional
    )
    
    print(f"Pipeline completed: {results['success']}")
    print(f"Final predictions: {results['results'].final_predictions}")
