import os
import cv2
import pandas as pd
import numpy as np
from deepforest import main
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box, Point, Polygon
import xml.etree.ElementTree as ET
from datetime import datetime
import json

# === LOAD CONFIGURATION FROM JSON ===
with open("scripts/train_process/validation_config.json", "r") as f:
    config = json.load(f)

location = config["location"]
epoch = config["epoch"]
model_path = config["model_path"].format(epoch=epoch)
image_path = config["image_path"].format(location=location)
label_path = config["label_path"].format(location=location)
output_path = config["output_path"].format(location=location, epoch=epoch)
confidence_threshold = config["confidence_threshold"]
iou_threshold = config["iou_threshold"]
patch_size = config["patch_size"]
patch_overlap = config["overlap_size"] / patch_size  # Convert overlap pixels to fraction
enable_postprocessing = config["enable_postprocessing"]
containment_threshold = config["containment_threshold"]
show_removed_boxes = config["show_removed_boxes"]
VALIDATION_METHOD = config["validation_method"]
kml_file = config["kml_file"]
VALIDATION_LOG_FILE = config["validation_log_file"]
enable_adaptive_confidence = config["enable_adaptive_confidence"]
adaptive_confidence_k = config["adaptive_confidence_k"]
iou_threshold_validation = config["iou_threshold_validation"]  # Read IoU threshold for validation from config

# === LOGGING CONFIGURATION ===
# VALIDATION_LOG_FILE = "validation_log.csv"

def extract_epoch_from_model_path(model_path):
    """Extract epoch number from model checkpoint filename."""
    import re
    # Look for patterns like "epoch_5", "epoch_180", etc.
    match = re.search(r'epoch_(\d+)', model_path)
    if match:
        return int(match.group(1))
    return None

def log_validation_run(location, epoch, image_path, label_path, confidence_threshold, 
                      iou_threshold, enable_postprocessing, containment_threshold,
                      patch_size, patch_overlap, confidence_stats, adaptive_stats,
                      metrics, postprocessing_stats, geofence_stats, output_path):
    """Log validation run parameters and results to CSV."""
    
    # Create log entry dictionary
    log_entry = {
        # Run identification
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'location': location,
        'epoch': epoch,
        'image_path': image_path,
        'label_path': label_path,
        'output_path': output_path,
        
        # Model parameters
        'confidence_threshold': confidence_threshold,
        'iou_threshold': iou_threshold,
        'patch_size': patch_size,
        'patch_overlap': patch_overlap,
        
        # Post-processing settings
        'enable_postprocessing': enable_postprocessing,
        'containment_threshold': containment_threshold if enable_postprocessing else None,
        
        # Confidence statistics
        'mean_confidence': confidence_stats.get('mean', 0),
        'std_confidence': confidence_stats.get('std', 0),
        'adaptive_threshold': confidence_stats.get('adaptive_threshold', 0),
        'adaptive_k_value': confidence_stats.get('k_value', 0),
        
        # Adaptive filtering results
        'predictions_before_adaptive': adaptive_stats.get('before', 0),
        'predictions_after_adaptive': adaptive_stats.get('after', 0),
        'adaptive_removed_count': adaptive_stats.get('removed', 0),
        'adaptive_percentage_kept': adaptive_stats.get('percentage_kept', 0),
        
        # Final metrics
        'ground_truth_trees': metrics.get('total_actual_trees', 0),
        'true_positives': metrics.get('tp', 0),
        'false_positives': metrics.get('fp', 0),
        'false_negatives': metrics.get('fn', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1_score': metrics.get('f1_score', 0),
        
        # Confidence analysis
        'avg_confidence_all': metrics.get('avg_confidence_all', 0),
        'std_confidence_all': metrics.get('std_confidence_all', 0),
        'avg_confidence_tp': metrics.get('avg_confidence_tp', 0),
        'std_confidence_tp': metrics.get('std_confidence_tp', 0),
        'avg_confidence_fp': metrics.get('avg_confidence_fp', 0),
        'std_confidence_fp': metrics.get('std_confidence_fp', 0),
        'tp_fp_confidence_diff': metrics.get('confidence_diff', 0),
        
        # Post-processing removal details
        'postprocess_removed_total': postprocessing_stats.get('total_removed', 0),
        'postprocess_removed_tp': postprocessing_stats.get('tp_removed', 0),
        'postprocess_removed_fp': postprocessing_stats.get('fp_removed', 0),
        'postprocess_incorrectly_removed': postprocessing_stats.get('incorrectly_removed', 0),
        
        # Geofence statistics
        'geofence_enabled': geofence_stats.get('enabled', False),
        'geofence_removed_predictions': geofence_stats.get('removed_predictions', 0),
        'geofence_excluded_ground_truth': geofence_stats.get('excluded_ground_truth', 0),
    }
    
    # Convert to DataFrame
    log_df = pd.DataFrame([log_entry])
    
    # Check if log file exists
    if os.path.exists(VALIDATION_LOG_FILE):
        # Append to existing file
        log_df.to_csv(VALIDATION_LOG_FILE, mode='a', header=False, index=False)
        print(f"📝 Validation run logged to {VALIDATION_LOG_FILE} (appended)")
    else:
        # Create new file with headers
        log_df.to_csv(VALIDATION_LOG_FILE, mode='w', header=True, index=False)
        print(f"📝 Created new validation log: {VALIDATION_LOG_FILE}")
    
    return log_entry

def analyze_removed_boxes_classification(removed_boxes, true_boxes, iou_threshold_validation=0.25):
    """Analyze removed boxes to classify them as TP or FP based on ground truth overlap."""
    tp_removed = 0
    fp_removed = 0
    incorrectly_removed_details = []
    
    for pred_box, center, confidence, row_data in removed_boxes:
        # Check if this removed box actually had good IoU with ground truth
        best_iou = 0
        best_match = None
        
        for i, (true_box, _) in enumerate(true_boxes):
            iou = calculate_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_match = i
        
        # Classify based on IoU with ground truth
        if best_iou >= iou_threshold_validation:
            tp_removed += 1
            incorrectly_removed_details.append({
                'confidence': confidence,
                'iou': best_iou,
                'ground_truth_index': best_match
            })
        else:
            fp_removed += 1
    
    return {
        'total_removed': len(removed_boxes),
        'tp_removed': tp_removed,
        'fp_removed': fp_removed,
        'incorrectly_removed': tp_removed,
        'incorrectly_removed_details': incorrectly_removed_details
    }


# Extract epoch from model path
epoch = extract_epoch_from_model_path(model_path)
if epoch is None:
    print(f"⚠️ Could not extract epoch from model path: {model_path}")
    epoch = "unknown"

print(f"📊 Validation Configuration:")
print(f"   Location: {location}")
print(f"   Epoch: {epoch}")
print(f"   Model: {model_path}")
print(f"   Image: {image_path}")
print(f"   Labels: {label_path}")

# === LOAD MODEL ===
print(f"Loading DeepForest model from: {model_path}")
model = main.deepforest.load_from_checkpoint(model_path)

# Configure patch parameters for large image processing
model.config["patch_size"] = patch_size  # From config
model.config["patch_overlap"] = patch_overlap  # From config
print(f"📐 Configured patch processing: {model.config['patch_size']}px with {model.config['patch_overlap']} overlap")

# === HELPER FUNCTIONS ===
def load_kml_polygon(kml_file_path):
    """Load polygon coordinates from KML file."""
    if not os.path.exists(kml_file_path):
        print(f"❌ KML file not found: {kml_file_path}")
        return None
    
    try:
        # Parse the KML file
        tree = ET.parse(kml_file_path)
        root = tree.getroot()
        
        # Handle KML namespace
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Look for coordinates in various KML structures
        coordinates_text = None
        
        # Try to find coordinates in Polygon > outerBoundaryIs > LinearRing > coordinates
        polygon_coords = root.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespace)
        if polygon_coords is not None:
            coordinates_text = polygon_coords.text
        else:
            # Try without namespace (some KML files don't use it)
            polygon_coords = root.find('.//coordinates')
            if polygon_coords is not None:
                coordinates_text = polygon_coords.text
        
        if coordinates_text is None:
            print(f"❌ No coordinates found in KML file: {kml_file_path}")
            return None
        
        # Parse coordinates (format: "lon,lat,alt lon,lat,alt ...")
        coord_points = []
        coordinates_text = coordinates_text.strip()
        
        for coord_str in coordinates_text.split():
            coord_parts = coord_str.split(',')
            if len(coord_parts) >= 2:
                try:
                    lon = float(coord_parts[0])
                    lat = float(coord_parts[1])
                    coord_points.append((lon, lat))
                except ValueError:
                    continue
        
        if len(coord_points) < 3:
            print(f"❌ Insufficient coordinates found in KML (need at least 3, found {len(coord_points)})")
            return None
        
        # Create Shapely polygon (note: Shapely uses (x, y) = (lon, lat))
        polygon = Polygon(coord_points)
        
        print(f"✅ Loaded KML geofence with {len(coord_points)} points")
        print(f"   Polygon bounds: {polygon.bounds}")
        
        return polygon
        
    except Exception as e:
        print(f"❌ Error parsing KML file: {e}")
        return None

def pixel_to_latlon(x, y, image_bounds, image_width, image_height):
    """Convert pixel coordinates to lat/lon given image bounds."""
    # Calculate lat/lon per pixel
    lon_per_pixel = (image_bounds['max_lon'] - image_bounds['min_lon']) / image_width
    lat_per_pixel = (image_bounds['max_lat'] - image_bounds['min_lat']) / image_height
    
    # Convert pixel to lat/lon
    # Note: y=0 is at the top of the image, but max_lat is at the top
    lon = image_bounds['min_lon'] + (x * lon_per_pixel)
    lat = image_bounds['max_lat'] - (y * lat_per_pixel)
    
    return lat, lon

def load_image_metadata(location):
    """Load image metadata to get geographic bounds."""
    # Try to load metadata file
    metadata_files = [
        f"data/{location}/metadata/{location}_metadata.json",
        f"data/{location}/metadata/{location}_highres_metadata.json",
        f"training_prepared/metadata/{location}_metadata.json"
    ]
    
    import json
    for metadata_file in metadata_files:
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Handle different metadata formats
                if 'bounds' in metadata and 'image_size' in metadata:
                    # Format 1: bounds dict with image_size
                    return {
                        'min_lon': metadata['bounds']['min_lon'],
                        'max_lon': metadata['bounds']['max_lon'], 
                        'min_lat': metadata['bounds']['min_lat'],
                        'max_lat': metadata['bounds']['max_lat']
                    }
                elif 'min_lon' in metadata and 'width_px' in metadata:
                    # Format 2: direct bounds with width_px/height_px
                    return {
                        'min_lon': metadata['min_lon'],
                        'max_lon': metadata['max_lon'],
                        'min_lat': metadata['min_lat'],
                        'max_lat': metadata['max_lat']
                    }
                    
            except Exception as e:
                print(f"⚠️ Error loading metadata from {metadata_file}: {e}")
                continue
    
    print(f"⚠️ No metadata found for {location}. Geofencing will be disabled.")
    return None

def filter_predictions_by_geofence(pred_boxes, geofence_polygon, image_bounds, image_width, image_height):
    """Filter predictions to only include those within the geofence polygon."""
    if geofence_polygon is None:
        return pred_boxes, []
    
    print(f"🌍 Applying geofence filtering...")
    
    filtered_boxes = []
    removed_boxes = []
    
    for pred_box, center, confidence, row_data in pred_boxes:
        # Convert center point from pixels to lat/lon
        center_x = (pred_box.bounds[0] + pred_box.bounds[2]) / 2
        center_y = (pred_box.bounds[1] + pred_box.bounds[3]) / 2
        
        lat, lon = pixel_to_latlon(center_x, center_y, image_bounds, image_width, image_height)
        
        # Create point in lat/lon coordinates (Shapely uses (x, y) = (lon, lat))
        geo_point = Point(lon, lat)
        
        # Check if point is within geofence
        if geofence_polygon.contains(geo_point):
            filtered_boxes.append((pred_box, center, confidence, row_data))
        else:
            removed_boxes.append((pred_box, center, confidence, row_data))
    
    removed_count = len(removed_boxes)
    print(f"   Geofence removed {removed_count} predictions outside boundary")
    print(f"   Remaining predictions: {len(filtered_boxes)}")
    
    return filtered_boxes, removed_boxes

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    # Get intersection area
    intersection = box1.intersection(box2).area
    
    # Get union area
    union = box1.union(box2).area
    
    # Calculate IoU
    if union == 0:
        return 0
    return intersection / union

def calculate_metrics(tp, fp, fn, confidences_all, confidences_tp, confidences_fp, stage_name):
    """Calculate and print metrics for a given stage"""
    total_actual_trees = tp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_actual_trees if total_actual_trees > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_confidence_all = np.mean(confidences_all) if confidences_all else 0.0
    avg_confidence_tp = np.mean(confidences_tp) if confidences_tp else 0.0
    avg_confidence_fp = np.mean(confidences_fp) if confidences_fp else 0.0
    
    std_confidence_all = np.std(confidences_all) if confidences_all else 0.0
    std_confidence_tp = np.std(confidences_tp) if confidences_tp else 0.0
    std_confidence_fp = np.std(confidences_fp) if confidences_fp else 0.0
    
    print(f"\n=== {stage_name.upper()} METRICS ===")
    print(f"Ground Truth Trees: {total_actual_trees}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"")
    print(f"Confidence Analysis:")
    print(f"  Average confidence (all predictions): {avg_confidence_all:.3f} ± {std_confidence_all:.3f}")
    print(f"  Average confidence (true positives):  {avg_confidence_tp:.3f} ± {std_confidence_tp:.3f}")
    print(f"  Average confidence (false positives): {avg_confidence_fp:.3f} ± {std_confidence_fp:.3f}")
    
    if confidences_tp and confidences_fp:
        confidence_diff = avg_confidence_tp - avg_confidence_fp
        print(f"  TP vs FP confidence difference: {confidence_diff:+.3f}")
        if confidence_diff > 0:
            print(f"  Model is more confident on correct predictions")
        else:
            print(f"  Model is more confident on incorrect predictions")
    
    return {
        'stage': stage_name,
        'total_actual_trees': total_actual_trees,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_confidence_all': avg_confidence_all,
        'avg_confidence_tp': avg_confidence_tp,
        'avg_confidence_fp': avg_confidence_fp,
        'std_confidence_all': std_confidence_all,
        'std_confidence_tp': std_confidence_tp,
        'std_confidence_fp': std_confidence_fp,
        'confidence_diff': avg_confidence_tp - avg_confidence_fp if (confidences_tp and confidences_fp) else 0
    }

def postprocess_merge_overlapping_boxes(pred_boxes, overlap_threshold=0.75):
    """
    Merge all boxes that overlap with a larger box by more than overlap_threshold (intersection/area of smaller box).
    All overlapping boxes are merged into a single new box, originals are deleted, and only the merged box remains.
    """
    if not pred_boxes:
        return pred_boxes, [], []

    print(f"🔄 Post-processing: Merging clusters of boxes with overlap > {overlap_threshold:.2f}...")

    merged_boxes = pred_boxes.copy()
    removed_boxes = []
    expanded_boxes = []
    i = 0
    while i < len(merged_boxes):
        boxA, centerA, confA, rowA = merged_boxes[i]
        # Find all boxes that overlap with boxA above threshold
        overlapping_indices = [i]
        for j in range(len(merged_boxes)):
            if j == i:
                continue
            boxB, centerB, confB, rowB = merged_boxes[j]
            intersection_area = boxA.intersection(boxB).area
            areaA = boxA.area
            areaB = boxB.area
            smaller_area = min(areaA, areaB)
            overlap = intersection_area / smaller_area if smaller_area > 0 else 0

            if overlap > overlap_threshold:
                overlapping_indices.append(j)
        # If more than one box overlaps, merge all
        if len(overlapping_indices) > 1:
            # Gather all boxes to merge
            boxes_to_merge = [merged_boxes[idx][0] for idx in overlapping_indices]
            centers_to_merge = [merged_boxes[idx][1] for idx in overlapping_indices]
            confs_to_merge = [merged_boxes[idx][2] for idx in overlapping_indices]
            rows_to_merge = [merged_boxes[idx][3] for idx in overlapping_indices]
            # Merge coordinates
            new_xmin = min([b.bounds[0] for b in boxes_to_merge])
            new_ymin = min([b.bounds[1] for b in boxes_to_merge])
            new_xmax = max([b.bounds[2] for b in boxes_to_merge])
            new_ymax = max([b.bounds[3] for b in boxes_to_merge])
            merged_box = box(new_xmin, new_ymin, new_xmax, new_ymax)
            merged_center = Point((new_xmin+new_xmax)/2, (new_ymin+new_ymax)/2)
            # Use confidence and row from largest box
            largest_idx = max(overlapping_indices, key=lambda idx: merged_boxes[idx][0].area)
            conf = merged_boxes[largest_idx][2]
            row = merged_boxes[largest_idx][3]
            print(f"   Merged boxes {overlapping_indices} into new box")
            # Add all originals to removed_boxes
            for idx in sorted(overlapping_indices, reverse=True):
                removed_boxes.append(merged_boxes[idx])
                del merged_boxes[idx]
            # Insert new merged box at position i
            merged_boxes.insert(i, (merged_box, merged_center, conf, row))
            expanded_boxes.append((merged_box, merged_center, conf, row))
            # Restart scan at i (since indices have changed)
            continue
        i += 1

    print(f"   Post-processing merged boxes. Final count: {len(merged_boxes)}")
    return merged_boxes, removed_boxes, expanded_boxes

# === LOAD GEOFENCE ===
geofence_polygon = None
image_bounds = None

if kml_file and os.path.exists(kml_file):
    print(f"\n{'='*60}")
    print("LOADING GEOFENCE")
    print(f"{'='*60}")
    
    # Load KML polygon
    geofence_polygon = load_kml_polygon(kml_file)
    
    # Load image metadata for coordinate conversion
    if geofence_polygon:
        image_bounds = load_image_metadata(location)
        if image_bounds:
            print(f"✅ Image bounds loaded: {image_bounds}")
        else:
            print(f"⚠️ Could not load image bounds - geofencing disabled")
            geofence_polygon = None
elif kml_file:
    print(f"⚠️ KML file specified but not found: {kml_file}")

# === LOAD IMAGE ===
print(f"Reading image: {image_path}")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image from: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]
print(f"Image size: {width}x{height}")

# === LOAD GROUND TRUTH LABELS ===
print(f"Reading labels from: {label_path}")
true_boxes_df = pd.read_csv(label_path)
print(f"Total ground truth boxes: {len(true_boxes_df)}")

true_boxes = []
ignored_boxes = []
geofence_filtered_labels = []
out_of_bounds_labels = []

for _, row in true_boxes_df.iterrows():
    bbox = box(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
    label = row.get("label", "")
    
    # Check if box is within image bounds
    if (row['xmin'] < 0 or row['ymin'] < 0 or 
        row['xmax'] >= width or row['ymax'] >= height or
        row['xmin'] >= row['xmax'] or row['ymin'] >= row['ymax']):
        out_of_bounds_labels.append((bbox, label, row))
        continue
    
    # Check if this label is within the geofence (if geofencing is enabled)
    if geofence_polygon and image_bounds:
        # Calculate center point of the label
        center_x = (row['xmin'] + row['xmax']) / 2
        center_y = (row['ymin'] + row['ymax']) / 2
        
        # Convert center point from pixels to lat/lon
        lat, lon = pixel_to_latlon(center_x, center_y, image_bounds, width, height)
        
        # Create point in lat/lon coordinates (Shapely uses (x, y) = (lon, lat))
        geo_point = Point(lon, lat)
        
        # Check if point is within geofence
        if not geofence_polygon.contains(geo_point):
            # Label is outside geofence - don't include it in calculations
            geofence_filtered_labels.append((bbox, label, row))
            continue
    
    if label == "Ignore":
        ignored_boxes.append(bbox)
    else:
        true_boxes.append((bbox, False))  # (shapely_box, is_matched)

# Print filtering results for ground truth
print(f"Ground truth labels loaded: {len(true_boxes_df)}")
print(f"Ground truth labels out of bounds: {len(out_of_bounds_labels)}")

if geofence_polygon and image_bounds:
    print(f"Ground truth labels filtered by geofence: {len(geofence_filtered_labels)}")
    print(f"Ground truth labels valid (within bounds and geofence): {len(true_boxes)}")
else:
    print(f"Ground truth labels valid (within image bounds): {len(true_boxes)}")

# Sample out-of-bounds labels for debugging
if out_of_bounds_labels:
    print(f"Sample out-of-bounds labels (first 3):")
    for i, (bbox, label, row) in enumerate(out_of_bounds_labels[:3]):
        print(f"  {i+1}. ({row['xmin']}, {row['ymin']}) to ({row['xmax']}, {row['ymax']}) - Image size: {width}x{height}")


# === RUN PREDICTIONS ===
print("\n" + "="*60)
print("RUNNING DEEPFOREST PREDICT_TILE")
print("="*60)
print(f"🔍 Using predict_tile with:")
print(f"   Patch size: {model.config['patch_size']}px")
print(f"   Patch overlap: {model.config['patch_overlap']}")
print(f"   IoU threshold: {iou_threshold}")
pred_df = model.predict_tile(
    image_path,  # Use the image path for predict_tile
    patch_size=model.config["patch_size"],
    patch_overlap=model.config["patch_overlap"],
    iou_threshold=iou_threshold
)

print(f"Total predictions before threshold: {len(pred_df) if pred_df is not None else 0}")

# Handle case where no predictions are made
if pred_df is None or len(pred_df) == 0:
    print("⚠️ No predictions returned by predict_tile")
    pred_df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label'])

pred_df = pred_df[pred_df["score"] > confidence_threshold]
print(f"Predictions after applying score > {confidence_threshold}: {len(pred_df)}")

# Convert predictions to our format
pred_boxes_deepforest = []
for _, row in pred_df.iterrows():
    pred_box = box(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
    center = Point((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2)
    confidence = row['score']

    # Skip predictions if center lies within any Ignore box
    if any(ignore_box.contains(center) for ignore_box in ignored_boxes):
        continue

    pred_boxes_deepforest.append((pred_box, center, confidence, row))

print(f"DeepForest predictions after filtering ignored areas: {len(pred_boxes_deepforest)}")

# === GEOFENCE FILTERING ===
geofence_removed_boxes = []
if geofence_polygon and image_bounds:
    print(f"\n{'='*60}")
    print("APPLYING GEOFENCE FILTERING")
    print(f"{'='*60}")
    
    pred_boxes_deepforest, geofence_removed_boxes = filter_predictions_by_geofence(
        pred_boxes_deepforest, geofence_polygon, image_bounds, width, height
    )
else:
    print(f"\n⚠️ Geofence filtering is disabled (no KML file or metadata)")

# === ADAPTIVE CONFIDENCE FILTERING (MOVED BEFORE POST-PROCESSING) ===
adaptive_removed_boxes = []  # Always define before filtering
confidence_stats = {}
adaptive_stats = {}

if enable_adaptive_confidence:
    print(f"\n{'='*60}")
    print("APPLYING ADAPTIVE CONFIDENCE FILTERING")
    print(f"{'='*60}")

    adaptive_removed_boxes = []
    confidence_stats = {}
    adaptive_stats = {}

    if pred_boxes_deepforest:
        # Extract all confidence values
        all_confidences = [confidence for _, _, confidence, _ in pred_boxes_deepforest]
        
        # Compute statistics
        mean_conf = np.mean(all_confidences)
        std_conf = np.std(all_confidences)
        k = adaptive_confidence_k  # Use config value for std multiplier
        
        # Calculate adaptive threshold
        adaptive_threshold = max(confidence_threshold, mean_conf - k * std_conf)
        
        # Store confidence statistics for logging
        confidence_stats = {
            'mean': mean_conf,
            'std': std_conf,
            'k_value': k,
            'adaptive_threshold': adaptive_threshold
        }
        
        # Print statistics
        print(f"📊 Confidence Statistics:")
        print(f"   Mean confidence (μ): {mean_conf:.3f}")
        print(f"   Standard deviation (σ): {std_conf:.3f}")
        print(f"   Using k = {k:.1f} standard deviations")
        print(f"   Adaptive threshold = max(0.25, μ - {k:.1f}σ) = max(0.25, {mean_conf:.3f} - {k:.1f}×{std_conf:.3f}) = {adaptive_threshold:.3f}")
        print(f"   Min confidence = μ - {k:.0f}σ = {mean_conf:.3f} - {k:.0f}×{std_conf:.3f} = {mean_conf - k * std_conf:.3f}")
        
        # Apply adaptive filtering BEFORE post-processing
        original_count = len(pred_boxes_deepforest)
        high_confidence_boxes = []
        
        for pred_box, center, confidence, row_data in pred_boxes_deepforest:
            if confidence >= adaptive_threshold:
                high_confidence_boxes.append((pred_box, center, confidence, row_data))
            else:
                adaptive_removed_boxes.append((pred_box, center, confidence, row_data))
        
        pred_boxes_deepforest = high_confidence_boxes
        filtered_count = len(pred_boxes_deepforest)
        removed_count = len(adaptive_removed_boxes)
        percentage_kept = (filtered_count/original_count)*100 if original_count > 0 else 0
        
        # Store adaptive filtering statistics for logging
        adaptive_stats = {
            'before': original_count,
            'after': filtered_count,
            'removed': removed_count,
            'percentage_kept': percentage_kept
        }
        
        print(f"📈 Adaptive Filtering Results:")
        print(f"   Predictions before adaptive filtering: {original_count}")
        print(f"   Predictions after adaptive filtering: {filtered_count}")
        print(f"   Predictions removed by adaptive filtering: {removed_count}")
        print(f"   Percentage kept: {percentage_kept:.1f}%")
    else:
        print(f"⚠️ No predictions available for adaptive filtering")
        adaptive_threshold = 0.25
        confidence_stats = {'mean': 0, 'std': 0, 'k_value': 2.0, 'adaptive_threshold': 0.25}
        adaptive_stats = {'before': 0, 'after': 0, 'removed': 0, 'percentage_kept': 0}
else:
    print(f"⚠️ Adaptive confidence filtering is disabled")
    adaptive_threshold = confidence_threshold
    confidence_stats = {'mean': 0, 'std': 0, 'k_value': 2.0, 'adaptive_threshold': adaptive_threshold}
    adaptive_stats = {'before': len(pred_boxes_deepforest), 'after': len(pred_boxes_deepforest), 'removed': 0, 'percentage_kept': 100}

# === POST-PROCESSING (NOW AFTER ADAPTIVE FILTERING) ===
removed_boxes = []  # Initialize empty list for removed boxes
postprocessing_stats = {}

if enable_postprocessing:
    print(f"\n{'='*60}")
    print("APPLYING POST-PROCESSING")
    print(f"{'='*60}")
    print(f"🔧 Post-processing {len(pred_boxes_deepforest)} high-confidence predictions...")
    pred_boxes_deepforest, removed_boxes, expanded_boxes = postprocess_merge_overlapping_boxes(
        pred_boxes_deepforest,
        overlap_threshold=containment_threshold
    )
    
    # Analyze removed boxes to classify as TP/FP
    postprocessing_stats = analyze_removed_boxes_classification(
        removed_boxes, true_boxes, iou_threshold_validation=0.25
    )
else:
    print(f"\n⚠️ Post-processing is disabled")
    postprocessing_stats = {
        'total_removed': 0,
        'tp_removed': 0,
        'fp_removed': 0,
        'incorrectly_removed': 0
    }

# === ANALYZE REMOVED BOXES FOR POTENTIAL ERRORS ===
print(f"\n🔍 Analyzing removed boxes for potential errors...")

if removed_boxes:
    # Print detailed analysis from postprocessing_stats
    print(f"📊 Post-processing removal analysis:")
    print(f"   Total boxes removed: {postprocessing_stats['total_removed']}")
    print(f"   True Positives removed: {postprocessing_stats['tp_removed']} (should have been kept)")
    print(f"   False Positives removed: {postprocessing_stats['fp_removed']} (correctly removed)")
    
    # Print detailed warnings for incorrectly removed boxes
    for detail in postprocessing_stats.get('incorrectly_removed_details', []):
        print(f"   ⚠️  WARNING: Removed box with confidence {detail['confidence']:.3f} had IoU {detail['iou']:.3f} with ground truth!")
    
    if postprocessing_stats['incorrectly_removed'] > 0:
        print(f"\n❌ CRITICAL ISSUE: {postprocessing_stats['incorrectly_removed']} correct detections were removed by post-processing!")
        print(f"   This creates artificial False Negatives and reduces recall.")
        print(f"   Consider:")
        print(f"   1. Raising containment_threshold from {containment_threshold} to 0.90-0.95")
        print(f"   2. Adding IoU check before removing overlapping boxes")
        print(f"   3. Keeping the higher-IoU box when both overlap with ground truth")
    else:
        print(f"   ✅ No incorrectly removed boxes detected")
else:
    print(f"   ℹ️  No boxes were removed by post-processing")

# === STEP 1: EVALUATE FINAL PREDICTIONS (AFTER POSTPROCESSING) ===
print("\n" + "="*60)
print("STEP 2: EVALUATING FINAL PREDICTIONS AFTER POSTPROCESSING")
print("="*60)

# Reset true_boxes for matching
true_boxes_reset = [(bbox, False) for bbox, _ in true_boxes]

tp_final, fp_final, fn_final = 0, 0, 0
matched_truth_ids_final = set()
final_confidences_all = []
final_confidences_tp = []
final_confidences_fp = []

print(f"Matching FINAL predictions to ground truth using {VALIDATION_METHOD.upper()} validation...")

if VALIDATION_METHOD == "containment":
    pred_matched = [False] * len(pred_boxes_deepforest)
    used_preds = set()  # Track already counted predictions

    for gt_idx, (true_box, matched) in enumerate(true_boxes_reset):
        candidates = []
        for pred_idx, (pred_box, center, confidence, row_data) in enumerate(pred_boxes_deepforest):
            if true_box.contains(center):
                iou = calculate_iou(pred_box, true_box)
                area = pred_box.area
                candidates.append((pred_idx, iou, confidence, area))

        if candidates:
            candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

            best_pred_idx = candidates[0][0]
            if best_pred_idx not in used_preds:
                pred_matched[best_pred_idx] = True
                true_boxes_reset[gt_idx] = (true_box, True)
                matched_truth_ids_final.add(gt_idx)
                tp_final += 1
                final_confidences_tp.append(pred_boxes_deepforest[best_pred_idx][2])
                used_preds.add(best_pred_idx)

            # All other candidates are FPs
            for cand in candidates[1:]:
                cand_idx = cand[0]
                if cand_idx not in used_preds:
                    fp_final += 1
                    final_confidences_fp.append(pred_boxes_deepforest[cand_idx][2])
                    used_preds.add(cand_idx)

    # Any predictions not assigned = FP
    for pred_idx, matched in enumerate(pred_matched):
        if not matched and pred_idx not in used_preds:
            fp_final += 1
            final_confidences_fp.append(pred_boxes_deepforest[pred_idx][2])
            used_preds.add(pred_idx)

else:
    for pred_box, center, confidence, row_data in pred_boxes_deepforest:
        final_confidences_all.append(confidence)
        best_match_index = None
        best_iou = 0
        for i, (true_box, matched) in enumerate(true_boxes_reset):
            if i not in matched_truth_ids_final:
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_index = i
        if best_match_index is not None and best_iou >= iou_threshold_validation:
            true_boxes_reset[best_match_index] = (true_boxes_reset[best_match_index][0], True)
            matched_truth_ids_final.add(best_match_index)
            tp_final += 1
            final_confidences_tp.append(confidence)
        else:
            fp_final += 1
            final_confidences_fp.append(confidence)

fn_final = sum(1 for _, matched in true_boxes_reset if not matched)

# DEBUG: Count and analyze false negatives
false_negative_boxes = []
for i, (true_box, matched) in enumerate(true_boxes_reset):
    if not matched:
        x1, y1, x2, y2 = map(int, true_box.bounds)
        false_negative_boxes.append((i, true_box, x1, y1, x2, y2))

print(f"\n🔍 DEBUG: False Negative Analysis")
print(f"   Total ground truth boxes in true_boxes_reset: {len(true_boxes_reset)}")
print(f"   Matched ground truth boxes: {sum(1 for _, matched in true_boxes_reset if matched)}")
print(f"   False negative count: {fn_final}")
print(f"   False negative boxes to be drawn: {len(false_negative_boxes)}")

if len(false_negative_boxes) != fn_final:
    print(f"   ⚠️  MISMATCH: FN count ({fn_final}) != FN boxes to draw ({len(false_negative_boxes)})")

# Sample some false negative coordinates for verification
if false_negative_boxes:
    print(f"   Sample false negative boxes (first 5):")
    for i, (idx, box, x1, y1, x2, y2) in enumerate(false_negative_boxes[:5]):
        print(f"     {i+1}. Index {idx}: ({x1}, {y1}) to ({x2}, {y2})")

# Calculate and display FINAL metrics
final_metrics = calculate_metrics(
    tp_final, fp_final, fn_final,
    final_confidences_all, final_confidences_tp, final_confidences_fp,
    "Final Predictions After Postprocessing"
)

# === DRAW BOXES BASED ON MATCHING LOGIC ===
print("\n" + "="*60)
print("CREATING VISUALIZATION (MATCHED LOGIC)")
print("="*60)

# Prepare lists for drawing
matched_pred_indices = set()
fp_pred_indices = set()
# For each ground truth, find all candidates and mark best as TP, others as FP
if VALIDATION_METHOD == "containment":
    pred_matched = [False] * len(pred_boxes_deepforest)
    pred_tp_indices = set()
    pred_fp_indices = set()
    for gt_idx, (true_box, matched) in enumerate(true_boxes_reset):
        candidates = []
        for pred_idx, (pred_box, center, confidence, row_data) in enumerate(pred_boxes_deepforest):
            if true_box.contains(center):
                iou = calculate_iou(pred_box, true_box)
                area = pred_box.area
                candidates.append((pred_idx, iou, confidence, area))
        if candidates:
            # Sort by iou desc, then confidence desc, then area desc
            candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            best_pred_idx = candidates[0][0]
            pred_tp_indices.add(best_pred_idx)
            pred_matched[best_pred_idx] = True
            true_boxes_reset[gt_idx] = (true_box, True)
            matched_pred_indices.add(best_pred_idx)
            # Mark all other candidates as FP
            for cand in candidates[1:]:
                pred_fp_indices.add(cand[0])
                fp_pred_indices.add(cand[0])
        # If no candidates, ground truth remains unmatched (FN)
    # Mark all unmatched predictions as FP
    for pred_idx, matched in enumerate(pred_matched):
        if not matched:
            pred_fp_indices.add(pred_idx)
            fp_pred_indices.add(pred_idx)
else:
    # IoU validation: prediction is TP if IoU >= threshold with any unmatched ground truth box
    pred_tp_indices = set()
    pred_fp_indices = set()
    for pred_idx, (pred_box, center, confidence, row_data) in enumerate(pred_boxes_deepforest):
        best_match_index = None
        best_iou = 0
        for i, (true_box, matched) in enumerate(true_boxes_reset):
            if i not in matched_truth_ids_final:
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_index = i
        if best_match_index is not None and best_iou >= iou_threshold_validation:
            true_boxes_reset[best_match_index] = (true_boxes_reset[best_match_index][0], True)
            matched_truth_ids_final.add(best_match_index)
            pred_tp_indices.add(pred_idx)
        else:
            pred_fp_indices.add(pred_idx)

# Draw predictions
for pred_idx, (pred_box, center, confidence, row_data) in enumerate(pred_boxes_deepforest):
    if pred_idx in pred_tp_indices:
        color = (0, 255, 0)  # Green for TP
    else:
        color = (0, 0, 255)  # Red for FP
    x1, y1, x2, y2 = map(int, pred_box.bounds)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # Draw confidence scores
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_y = max(y1 - 10, 15)
    df_text = f"{confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(df_text, font, font_scale, font_thickness)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1 - 2, text_y - text_height - 2), (x1 + text_width + 2, text_y + baseline + 2), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, df_text, (x1, text_y), font, font_scale, (0, 0, 255), font_thickness)

# Draw False Negatives (missed ground truth)
blue_boxes_drawn = 0
invalid_boxes = []
for i, (true_box, matched) in enumerate(true_boxes_reset):
    if not matched:
        x1, y1, x2, y2 = map(int, true_box.bounds)
        
        # Check if coordinates are valid
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2:
            invalid_boxes.append((i, x1, y1, x2, y2, "Invalid coordinates"))
            continue
        
        # Check if box is within image bounds
        if x1 >= image.shape[1] or y1 >= image.shape[0] or x2 <= 0 or y2 <= 0:
            invalid_boxes.append((i, x1, y1, x2, y2, "Outside image bounds"))
            continue
            
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        blue_boxes_drawn += 1

print(f"\n🔍 VISUALIZATION DEBUG:")
print(f"   Blue boxes (false negatives) drawn: {blue_boxes_drawn}")
print(f"   Calculated false negatives: {fn_final}")
print(f"   Invalid/undrawn boxes: {len(invalid_boxes)}")

if len(invalid_boxes) > 0:
    print(f"   Invalid boxes details:")
    for i, x1, y1, x2, y2, reason in invalid_boxes:
        print(f"     Index {i}: ({x1}, {y1}) to ({x2}, {y2}) - {reason}")

if blue_boxes_drawn != fn_final:
    print(f"   ⚠️  MISMATCH: Blue boxes drawn ({blue_boxes_drawn}) != FN calculated ({fn_final})")
    print(f"   Missing boxes: {fn_final - blue_boxes_drawn}")
else:
    print(f"   ✅ Blue boxes drawn matches FN count")

# Draw Geofence Removed Boxes
if show_removed_boxes and geofence_removed_boxes:
    print(f"Drawing {len(geofence_removed_boxes)} geofence-removed boxes in orange...")
    for pred_box, center, confidence, row_data in geofence_removed_boxes:
        # ORANGE box: Removed by geofence filtering
        x1, y1, x2, y2 = map(int, pred_box.bounds)
        
        # Draw orange bounding box 
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 1)  # Orange color in BGR
        
        # Draw confidence scores with orange text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        
        # Position text above the box
        text_y = max(y1 - 5, 10)
        
        # Draw geofence removed box confidence with orange background
        geofence_text = f"G:{confidence:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(geofence_text, font, font_scale, font_thickness)
        
        # Create overlay for transparency
        overlay = image.copy()
        cv2.rectangle(overlay, (x1 - 1, text_y - text_height - 1), (x1 + text_width + 1, text_y + baseline + 1), (200, 220, 255), -1)
        
        # Apply transparency
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw the text in dark orange
        cv2.putText(image, geofence_text, (x1, text_y), font, font_scale, (0, 100, 200), font_thickness)

# Draw Adaptive Filtering Removed Boxes
if show_removed_boxes and adaptive_removed_boxes:
    print(f"Drawing {len(adaptive_removed_boxes)} adaptive-filtered boxes in purple...")
    for pred_box, center, confidence, row_data in adaptive_removed_boxes:
        # PURPLE box: Removed by adaptive confidence filtering
        x1, y1, x2, y2 = map(int, pred_box.bounds)
        
        # Draw purple bounding box 
        cv2.rectangle(image, (x1, y1), (x2, y2), (128, 0, 128), 1)  # Purple color in BGR
        
        # Draw confidence scores with purple text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        
        # Position text above the box
        text_y = max(y1 - 5, 10)
        
        # Draw adaptive removed box confidence with purple background
        adaptive_text = f"A:{confidence:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(adaptive_text, font, font_scale, font_thickness)
        
        # Create overlay for transparency
        overlay = image.copy()
        cv2.rectangle(overlay, (x1 - 1, text_y - text_height - 1), (x1 + text_width + 1, text_y + baseline + 1), (200, 150, 200), -1)
        
        # Apply transparency
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw the text in dark purple
        cv2.putText(image, adaptive_text, (x1, text_y), font, font_scale, (64, 0, 64), font_thickness)

# Draw removed postprocessed boxes in thin light gray ONLY if show_removed_boxes is True
if show_removed_boxes and enable_postprocessing and removed_boxes:
    print(f"Drawing {len(removed_boxes)} postprocessed removed boxes in light gray...")
    for pred_box, center, confidence, row_data in removed_boxes:
        x1, y1, x2, y2 = map(int, pred_box.bounds)
        cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), 1)  # Light gray, thin line
        # Optionally, draw confidence score in gray
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        text_y = max(y1 - 5, 10)
        removed_text = f"R:{confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(removed_text, font, font_scale, font_thickness)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1 - 1, text_y - text_height - 1), (x1 + text_width + 1, text_y + baseline + 1), (220, 220, 220), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.putText(image, removed_text, (x1, text_y), font, font_scale, (100, 100, 100), font_thickness)

# === SAVE OUTPUT ===
output_dir = os.path.dirname(output_path)
if output_dir:  # Only create directory if output_path includes a directory
    os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(output_path, image)
print(f"\nOutput saved to {output_path}")

# === FINAL SUMMARY ===
print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)
print(f"Mode: DeepForest Only")
print(f"DeepForest confidence threshold: {confidence_threshold}")
print(f"Prediction IoU threshold: {iou_threshold}")
print(f"Validation IoU threshold: {iou_threshold_validation}")
print(f"Geofence filtering: {'Enabled' if geofence_polygon else 'Disabled'}")
if geofence_polygon:
    print(f"KML file: {kml_file}")
    if geofence_removed_boxes:
        print(f"Geofence removed predictions: {len(geofence_removed_boxes)}")
    if 'geofence_filtered_labels' in locals() and geofence_filtered_labels:
        print(f"Geofence excluded ground truth labels: {len(geofence_filtered_labels)}")
        print(f"Ground truth labels used in validation: {len(true_boxes)}")
print(f"Post-processing enabled: {enable_postprocessing}")
if enable_postprocessing:
    print(f"Containment threshold: {containment_threshold} ({containment_threshold*100:.0f}%)")
    print(f"Show removed boxes: {show_removed_boxes}")
    if removed_boxes:
        print(f"Post-processing removed boxes: {len(removed_boxes)}")
print(f"Adaptive filtering threshold: {adaptive_threshold:.3f}")
if adaptive_removed_boxes:
    print(f"Adaptive filtering removed boxes: {len(adaptive_removed_boxes)}")
print(f"Output image: {output_path}")
print("\nBox Color Legend:")
print("  🟢 Green: True Positive (correct detection)")
print("  🔴 Red: False Positive (incorrect detection)")
print("  🔵 Blue: False Negative (missed ground truth)")
if geofence_removed_boxes:
    print("  🟠 Orange: Removed by geofence (G:confidence)")
if adaptive_removed_boxes:
    print("  🟣 Purple: Removed by adaptive filtering (A:confidence)")
    
print(f"\n📊 Processing Order (FIXED):")
print(f"   1. Initial predictions: {len(pred_df) if 'pred_df' in locals() else 'N/A'}")
print(f"   2. Basic confidence filter (>{confidence_threshold}): Applied")
print(f"   3. Ignore area filtering: Applied")
if geofence_polygon:
    print(f"   4. Geofence filtering: Removed {len(geofence_removed_boxes)}")
if adaptive_removed_boxes:
    print(f"   5. Adaptive confidence filtering: Removed {len(adaptive_removed_boxes)}")
if enable_postprocessing:
    print(f"   6. Post-processing (containment): Removed {len(removed_boxes)}")
print(f"   7. Final predictions: {len(pred_boxes_deepforest)}")

if postprocessing_stats.get('incorrectly_removed', 0) > 0:
    print(f"\n❌ ISSUES DETECTED:")
    print(f"   {postprocessing_stats['incorrectly_removed']} correct detections were removed by post-processing")
    print(f"   This indicates the containment threshold ({containment_threshold}) may be too aggressive")

# === LOG VALIDATION RUN ===
print(f"\n📝 Logging validation run...")

# Prepare geofence statistics
geofence_stats = {
    'enabled': geofence_polygon is not None,
    'removed_predictions': len(geofence_removed_boxes) if geofence_removed_boxes else 0,
    'excluded_ground_truth': len(geofence_filtered_labels) if 'geofence_filtered_labels' in locals() and geofence_filtered_labels else 0
}

# Log the validation run
log_entry = log_validation_run(
    location=location,
    epoch=epoch,
    image_path=image_path,
    label_path=label_path,
    confidence_threshold=confidence_threshold,
    iou_threshold=iou_threshold,
    enable_postprocessing=enable_postprocessing,
    containment_threshold=containment_threshold,
    patch_size=model.config["patch_size"],
    patch_overlap=model.config["patch_overlap"],
    confidence_stats=confidence_stats,
    adaptive_stats=adaptive_stats,
    metrics=final_metrics,
    postprocessing_stats=postprocessing_stats,
    geofence_stats=geofence_stats,
    output_path=output_path
)
