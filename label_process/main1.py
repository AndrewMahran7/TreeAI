import os
import math
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from deepforest import main as deepforest_main
import simplekml
import cv2
from glob import glob
from pykml import parser
from lxml import etree
from pathlib import Path
import pyproj  # Added for coordinate conversion


def predict_from_jpg(jpg_path, model_path, output_image, output_pred_csv, output_kml,
                     json_metadata_path, geofence_coords,
                     patch_size=800, patch_overlap=0.25, iou_threshold=0.25, conf_min=0.25):
    """
    Run tree detection with DeepForest model and apply geofence filtering.
    Similar to test.py but with KML geofence functionality.
    
    Args:
        jpg_path (str): Path to input image
        model_path (str): Path to trained model checkpoint or "1.5.0" for release model
        output_image (str): Path to save output image with predictions
        output_pred_csv (str): Path to save predictions CSV
        output_kml (str): Path to save KML file with tree points
        json_metadata_path (str): Path to image metadata JSON for coordinate conversion
        geofence_coords (list): List of (lon, lat) coordinates defining geofence polygon
        patch_size (int): Size of patches for tiled prediction (default: 400)
        patch_overlap (float): Overlap between patches 0.0-1.0 (default: 0.25)
        iou_threshold (float): IoU threshold for NMS (default: 0.25)
        conf_min (float): Minimum confidence threshold (default: 0.25)
    
    Returns:
        tuple: (predictions_dataframe, output_image_array, metrics_dict)
    """

    print("Loading DeepForest model...")
    if model_path == "1.5.0":
        # Use DeepForest release model
        model = deepforest_main.deepforest()
        model.use_release()
        print("Loaded DeepForest release model 1.5.0")
    else:
        # Use custom checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = deepforest_main.deepforest.load_from_checkpoint(model_path)
        print(f"Loaded custom model from checkpoint")

    # === LOAD IMAGE ===
    print(f"Reading image: {jpg_path}")
    if not os.path.exists(jpg_path):
        raise FileNotFoundError(f"Image not found: {jpg_path}")
    
    # Try to load with PIL first to handle very large images
    try:
        print("Attempting to load image with PIL...")
        
        # Temporarily increase PIL's decompression bomb limit for legitimate large images
        from PIL import Image
        original_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None  # Remove the limit temporarily
        
        pil_img = Image.open(jpg_path)
        width, height = pil_img.size
        print(f"Image size: {width}x{height} ({width * height:,} pixels)")
        
        # Restore original PIL limit
        Image.MAX_IMAGE_PIXELS = original_max_pixels
        
        # Check if image is too large for OpenCV
        max_opencv_pixels = 2**30  # ~1 billion pixels (OpenCV limit is around here)
        total_pixels = width * height
        
        if total_pixels > max_opencv_pixels:
            print(f"⚠️  Warning: Image is very large ({total_pixels:,} pixels)")
            print(f"   This may cause memory issues or processing failures.")
            print(f"   Consider processing smaller tiles instead of the full image.")
            print(f"   Estimated memory usage: ~{(total_pixels * 3) / (1024**3):.1f} GB")
            
            # For very large images, we'll still try but may need to handle memory carefully
            # Convert PIL to OpenCV format manually to avoid cv2.imread limits
            print("Converting large image from PIL to OpenCV format...")
            pil_img_rgb = pil_img.convert('RGB')
            img_array = np.array(pil_img_rgb)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            pil_img.close()  # Free memory
            pil_img_rgb = None  # Free memory
            img_array = None   # Free memory
            print("Successfully converted large image from PIL to OpenCV format")
        else:
            # For smaller images, use OpenCV directly
            pil_img.close()
            img = cv2.imread(jpg_path)
            if img is None:
                raise ValueError(f"Could not load image with OpenCV: {jpg_path}")
            height, width = img.shape[:2]
            print("Loaded image with OpenCV")
            
    except Exception as e:
        print(f"Error loading image: {e}")
        print(f"This image is extremely large ({width*height:,} pixels if loaded)")
        print(f"For images this size, consider:")
        print(f"  1. Processing smaller tiles instead of the full image")
        print(f"  2. Using a machine with more RAM (estimated need: {(width*height*3)/(1024**3):.1f} GB)")
        print(f"  3. Reducing the image size before processing")
        raise ValueError(f"Could not load image from: {jpg_path}")
    
    print(f"Final image dimensions: {width}x{height}")

    print("Running tree crown detection...")
    print(f"  Patch size: {patch_size}")
    print(f"  Patch overlap: {patch_overlap}")
    print(f"  IoU threshold: {iou_threshold}")
    
    # For very large images, suggest smaller patch sizes to reduce memory usage
    if width * height > 100_000_000:  # > 100 million pixels
        if patch_size > 800:
            print(f"  ⚠️  Note: Large image detected. Consider using smaller patch_size (≤800) to reduce memory usage.")
        print(f"  💡 Tip: For faster processing, consider splitting this image into smaller tiles first.")
    
    predicted = model.predict_tile(
        jpg_path, 
        patch_size=patch_size, 
        patch_overlap=patch_overlap, 
        iou_threshold=iou_threshold
    )
    

    # Capture raw prediction count before any filtering
    total_raw_predictions = len(predicted)
    print(f"Total predictions before filtering: {total_raw_predictions}")
    
    # Apply aspect ratio filtering
    predicted["width"] = predicted["xmax"] - predicted["xmin"]
    predicted["height"] = predicted["ymax"] - predicted["ymin"]
    predicted = predicted[(predicted["width"] / predicted["height"]).between(0.4, 2.5)]
    after_aspect_ratio = len(predicted)
    aspect_ratio_filtered = total_raw_predictions - after_aspect_ratio
    print(f"After aspect ratio filtering: {after_aspect_ratio} (removed {aspect_ratio_filtered})")

    # === LOAD METADATA ===
    print(f"Loading metadata: {json_metadata_path}")
    if not os.path.exists(json_metadata_path):
        raise FileNotFoundError(f"Metadata JSON file not found: {json_metadata_path}")
    
    with open(json_metadata_path, "r") as f:
        meta = json.load(f)
    
    # Extract bounds from metadata JSON - handle multiple formats
    bounds = None
    if "bounds" in meta:
        # Original format: bounds at root level
        bounds = meta["bounds"]
        print("Using bounds from root level (original format)")
    elif "geographic_info" in meta and "bounds" in meta["geographic_info"]:
        # Tile format: bounds nested under geographic_info
        bounds = meta["geographic_info"]["bounds"]
        print("Using bounds from geographic_info (tile format)")
    else:
        raise ValueError(f"Metadata JSON missing 'bounds' field in expected locations: {json_metadata_path}")
    
    min_lat = bounds["min_lat"]
    max_lat = bounds["max_lat"]
    min_lon = bounds["min_lon"]
    max_lon = bounds["max_lon"]
    
    # Get image dimensions from metadata or image file - handle multiple formats
    width_px = height_px = None
    
    if "image_size" in meta:
        # Original format: image_size at root level
        width_px = meta["image_size"]["width"]
        height_px = meta["image_size"]["height"]
        print(f"Using image size from root level: {width_px}x{height_px}")
    elif "image_properties" in meta and "size" in meta["image_properties"]:
        # Tile format: size nested under image_properties
        width_px = meta["image_properties"]["size"]["width"]
        height_px = meta["image_properties"]["size"]["height"]
        print(f"Using image size from image_properties: {width_px}x{height_px}")
    elif "width_px" in meta and "height_px" in meta:
        # Legacy format: width_px/height_px at root level
        width_px = meta["width_px"]
        height_px = meta["height_px"]
        print(f"Using legacy image size format: {width_px}x{height_px}")
    else:
        # Fall back to reading from actual image file
        width_px = width
        height_px = height
        print(f"Using image size from image file: {width_px}x{height_px}")
    
    print(f"Image bounds: {min_lat:.6f}, {min_lon:.6f} → {max_lat:.6f}, {max_lon:.6f}")

    def xy_to_latlon(row):
        # Calculate center point of the bounding box (same as test.py)
        x_center = (row["xmin"] + row["xmax"]) / 2
        y_center = (row["ymin"] + row["ymax"]) / 2
        # Convert pixel coordinates to lat/lon using image bounds
        lat = max_lat - (y_center / height_px) * (max_lat - min_lat)
        lon = min_lon + (x_center / width_px) * (max_lon - min_lon)
        return lat, lon

    def xy_to_latlon_bbox(row):
        # Convert all four corners of bounding box to lat/lon
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        
        # Convert each corner
        lat_min = max_lat - (y2 / height_px) * (max_lat - min_lat)  # bottom edge (ymax -> lat_min)
        lat_max = max_lat - (y1 / height_px) * (max_lat - min_lat)  # top edge (ymin -> lat_max)
        lon_min = min_lon + (x1 / width_px) * (max_lon - min_lon)  # left edge
        lon_max = min_lon + (x2 / width_px) * (max_lon - min_lon)  # right edge
        
        return lat_min, lat_max, lon_min, lon_max

    print("Converting coordinates to lat/lon...")
    latlons = predicted.apply(xy_to_latlon, axis=1, result_type="expand")
    predicted["lat"] = latlons[0]
    predicted["lon"] = latlons[1]
    
    # Add bounding box lat/lon coordinates
    bbox_latlons = predicted.apply(xy_to_latlon_bbox, axis=1, result_type="expand")
    predicted["lat_min"] = bbox_latlons[0]
    predicted["lat_max"] = bbox_latlons[1]
    predicted["lon_min"] = bbox_latlons[2]
    predicted["lon_max"] = bbox_latlons[3]

    # Apply confidence and geofence filtering
    geofence = Polygon(geofence_coords)
    results = []
    filtered_by_confidence = 0
    filtered_by_geofence = 0
    invalid_coordinates = 0

    for idx, row in predicted.iterrows():
        # Skip if confidence too low
        if row["score"] < conf_min:
            filtered_by_confidence += 1
            continue
            
        # Skip if coordinates invalid
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            invalid_coordinates += 1
            continue

        # Skip if outside geofence
        point = Point(row["lon"], row["lat"])
        if not geofence.contains(point):
            filtered_by_geofence += 1
            continue

        results.append(row.to_dict())

    print(f"Filtered by confidence < {conf_min}: {filtered_by_confidence}")
    print(f"Filtered by invalid coordinates: {invalid_coordinates}")
    print(f"Filtered by geofence: {filtered_by_geofence}")
    print(f"Final predictions: {len(results)}")

    # Convert to DataFrame
    if len(results) > 0:
        pred_df_filtered = pd.DataFrame(results)
    else:
        pred_df_filtered = pd.DataFrame()
        print("WARNING: No predictions passed all filters")

    # === CALCULATE CONFIDENCE STATISTICS ===
    if len(pred_df_filtered) > 0:
        avg_confidence = pred_df_filtered["score"].mean()
        max_confidence = pred_df_filtered["score"].max()
        min_confidence = pred_df_filtered["score"].min()
        
        print(f"Confidence Statistics:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Maximum confidence: {max_confidence:.3f}")
        print(f"  Minimum confidence: {min_confidence:.3f}")
        print(f"  Confidence threshold: {conf_min}")
    else:
        avg_confidence = max_confidence = min_confidence = 0.0

    # === VISUALIZATION ===
    print("Drawing prediction boxes on image...")
    output_img = img.copy()

    # Initialize KML
    kml = simplekml.Kml()

    for _, row in pred_df_filtered.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['score']
        
        # Calculate center for KML point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw green bounding box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.circle(output_img, (cx, cy), 4, (0, 255, 0), -1)

        # # Draw confidence score above the box (similar to test.py)
        # confidence_text = f"{confidence:.3f}"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.6
        # font_thickness = 2
        # 
        # # Get text size to position it properly
        # (text_width, text_height), baseline = cv2.getTextSize(confidence_text, font, font_scale, font_thickness)
        # 
        # # Position text above the box
        # text_x = x1
        # text_y = max(y1 - 10, text_height + 10)
        # 
        # # Draw white background for text readability
        # cv2.rectangle(output_img, 
        #               (text_x - 2, text_y - text_height - 2), 
        #               (text_x + text_width + 2, text_y + baseline + 2), 
        #               (255, 255, 255), -1)
        # 
        # # Draw confidence text in black
        # cv2.putText(output_img, confidence_text, (text_x, text_y), 
        #             font, font_scale, (0, 0, 0), font_thickness)

        # Add to KML
        label = f"Tree - {int(confidence * 100)}%"
        point = kml.newpoint(name=label, coords=[(row["lon"], row["lat"])])

        style = simplekml.Style()
        style.iconstyle.scale = 1.2
        style.iconstyle.color = simplekml.Color.green
        style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/trees.png"
        point.style = style

    # === SAVE OUTPUTS ===
    print("Saving outputs...")
    
    # Save output image
    cv2.imwrite(output_image, output_img)
    print(f"Saved prediction image to {output_image}")
    
    # Save KML
    kml.save(output_kml)
    print(f"Saved KML to {output_kml}")

    # Save CSV with additional columns (similar to test.py)
    if len(pred_df_filtered) > 0:
        # Add geometry column
        def create_geometry(row):
            x2, y2, x1, y1 = row['xmax'], row['ymax'], row['xmin'], row['ymin']
            return f"POLYGON (({x2} {y1}, {x2} {y2}, {x1} {y2}, {x1} {y1}, {x2} {y1}))"
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            return f"POLYGON (({x2} {y1}, {x2} {y2}, {x1} {y2}, {x1} {y1}, {x2} {y1}))"
        
        pred_df_filtered['geometry'] = pred_df_filtered.apply(create_geometry, axis=1)
        
        # Round confidence scores to 3 decimal places for CSV
        pred_df_filtered['score'] = pred_df_filtered['score'].round(3)
        
        # Ensure consistent column order and format
        output_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_path', 'geometry', 
                         'lat', 'lon', 'lat_min', 'lat_max', 'lon_min', 'lon_max']
        
        # Add missing columns if needed
        if 'image_path' not in pred_df_filtered.columns:
            pred_df_filtered['image_path'] = Path(jpg_path).name
            
        # Reorder columns
        pred_df_filtered = pred_df_filtered[[col for col in output_columns if col in pred_df_filtered.columns]]
        
        pred_df_filtered.to_csv(output_pred_csv, index=False)
        print(f"Prediction CSV saved to {output_pred_csv}")
    else:
        # Save empty CSV with headers
        empty_df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_path', 
                                        'geometry', 'lat', 'lon', 'lat_min', 'lat_max', 'lon_min', 'lon_max'])
        empty_df.to_csv(output_pred_csv, index=False)
        print(f"Empty prediction CSV saved to {output_pred_csv}")

    # === PREPARE METRICS ===
    metrics = {
        'total_raw_predictions': total_raw_predictions,
        'after_aspect_ratio_filter': after_aspect_ratio,
        'aspect_ratio_filtered': aspect_ratio_filtered,
        'confidence_filtered': filtered_by_confidence,
        'invalid_coordinates': invalid_coordinates,
        'geofence_filtered': filtered_by_geofence,
        'final_predictions': len(pred_df_filtered),
        'confidence_threshold': conf_min,
        'avg_confidence': avg_confidence,
        'max_confidence': max_confidence,
        'min_confidence': min_confidence,
        'image_size': (width, height)
    }
    
    return pred_df_filtered, output_img, metrics





def generate_confidence_histogram(predictions_df, output_path, location, metrics):
    """
    Generate and save a histogram plot of prediction confidences.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions with 'score' column
        output_path (str): Path to save the histogram image
        location (str): Location name for the title
        metrics (dict): Metrics dictionary containing filtering information
    """
    if predictions_df.empty or 'score' not in predictions_df.columns:
        print("   No predictions to create histogram")
        return
    
    confidences = predictions_df['score'].values
    
    # Create the histogram plot
    plt.figure(figsize=(10, 6))
    
    # Create histogram with bins
    n_bins = min(30, len(confidences))  # Limit bins to avoid sparse histograms
    counts, bins, patches = plt.hist(confidences, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    median_conf = np.median(confidences)
    
    # Add vertical lines for statistics
    plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
    plt.axvline(median_conf, color='green', linestyle='--', linewidth=2, label=f'Median: {median_conf:.3f}')
    plt.axvline(metrics['confidence_threshold'], color='orange', linestyle='-', linewidth=2, 
                label=f'Threshold: {metrics["confidence_threshold"]:.3f}')
    
    # Formatting
    plt.title(f'Prediction Confidence Distribution - {location}\n'
              f'Final Predictions: {len(confidences)} | Mean: {mean_conf:.3f} ± {std_conf:.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with summary statistics
    textstr = f'Statistics:\n' \
              f'Count: {len(confidences)}\n' \
              f'Mean: {mean_conf:.3f}\n' \
              f'Std: {std_conf:.3f}\n' \
              f'Min: {np.min(confidences):.3f}\n' \
              f'Max: {np.max(confidences):.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Confidence histogram saved to {output_path}")


def process_image_with_geofence(jpg_path, json_metadata_path, kml_path, model_path, output_dir, 
                               patch_size=400, patch_overlap=0.25, iou_threshold=0.25, conf_min=0.25):
    """
    Complete workflow: JSON metadata->Predictions->Geofence->Output
    
    Args:
        jpg_path (str): Path to input image
        json_metadata_path (str): Path to JSON metadata file
        kml_path (str): Path to KML geofence file
        model_path (str): Path to trained model checkpoint or "1.5.0" for release model
        output_dir (str): Directory to save all outputs
        patch_size (int): Size of patches for tiled prediction (default: 400)
        patch_overlap (float): Overlap between patches 0.0-1.0 (default: 0.25)
        iou_threshold (float): IoU threshold for NMS (default: 0.25)
        conf_min (float): Minimum confidence threshold (default: 0.25)
    
    Returns:
        dict: Complete results including paths and metrics
    """
    
    # === SETUP PATHS ===
    image_name = Path(jpg_path).stem
    location = image_name  # Use image name as location
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    output_image = os.path.join(output_dir, f"{image_name}_predictions.jpg")
    output_pred_csv = os.path.join(output_dir, f"{image_name}_predictions.csv")
    output_kml = os.path.join(output_dir, f"{image_name}_predictions.kml")
    
    print(f"Processing image: {image_name}")
    print(f"Output directory: {output_dir}")
    
    # === STEP 1: LOAD METADATA ===
    print("Step 1: Loading JSON metadata...")
    if not os.path.exists(json_metadata_path):
        raise FileNotFoundError(f"JSON metadata file not found: {json_metadata_path}")
    
    with open(json_metadata_path, "r") as f:
        metadata = json.load(f)
    
    # === STEP 2: LOAD GEOFENCE ===
    print("Step 2: Loading geofence from KML...")
    if not os.path.exists(kml_path):
        raise FileNotFoundError(f"KML geofence file not found: {kml_path}")
    
    with open(kml_path, "r", encoding="utf-8") as file:
        root = parser.parse(file).getroot()
    
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    polygon = root.xpath(".//kml:Polygon", namespaces=namespace)[0]
    coords_text = polygon.xpath(".//kml:coordinates", namespaces=namespace)[0].text.strip()
    geofence_coords = [tuple(map(float, c.split(",")))[:2] for c in coords_text.split() if c.strip()]
    
    print(f"   Geofence has {len(geofence_coords)} vertices")
    
    # === STEP 3: RUN PREDICTIONS WITH GEOFENCE ===
    print("Step 3: Running tree detection with geofence filtering...")
    predictions, output_img, metrics = predict_from_jpg(
        jpg_path=jpg_path,
        model_path=model_path,
        output_image=output_image,
        output_pred_csv=output_pred_csv,
        output_kml=output_kml,
        json_metadata_path=json_metadata_path,
        geofence_coords=geofence_coords,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        iou_threshold=iou_threshold,
        conf_min=conf_min
    )

    # === STEP 4: GENERATE CONFIDENCE HISTOGRAM ===
    print("Step 4: Generating confidence histogram...")
    histogram_path = os.path.join(output_dir, f"{image_name}_confidence_histogram.png")
    generate_confidence_histogram(predictions, histogram_path, location, metrics)
    
    # === COMPILE RESULTS ===
    results = {
        'location': location,
        'input_files': {
            'image': jpg_path,
            'metadata': json_metadata_path,
            'kml': kml_path,
            'model': model_path
        },
        'output_files': {
            'predictions_image': output_image,
            'predictions_csv': output_pred_csv,
            'predictions_kml': output_kml,
            'confidence_histogram': histogram_path
        },
        'metadata': metadata,
        'predictions': predictions,
        'metrics': metrics,
        'geofence_vertices': len(geofence_coords)
    }
    
    return results


def process_image_without_geofence(jpg_path, json_metadata_path, model_path, output_dir, 
                                   patch_size=400, patch_overlap=0.25, iou_threshold=0.25, conf_min=0.25):
    """
    Complete workflow without geofence filtering: JSON metadata->Predictions->Output
    
    Args:
        jpg_path (str): Path to input image
        json_metadata_path (str): Path to JSON metadata file
        model_path (str): Path to trained model checkpoint or "1.5.0" for release model
        output_dir (str): Directory to save all outputs
        patch_size (int): Size of patches for tiled prediction (default: 400)
        patch_overlap (float): Overlap between patches 0.0-1.0 (default: 0.25)
        iou_threshold (float): IoU threshold for NMS (default: 0.25)
        conf_min (float): Minimum confidence threshold (default: 0.25)
    
    Returns:
        dict: Complete results including paths and metrics
    """
    
    # === SETUP PATHS ===
    image_name = Path(jpg_path).stem
    location = image_name  # Use image name as location
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    output_image = os.path.join(output_dir, f"{image_name}_predictions.jpg")
    output_pred_csv = os.path.join(output_dir, f"{image_name}_predictions.csv")
    output_kml = os.path.join(output_dir, f"{image_name}_predictions.kml")
    
    print(f"Processing image: {image_name}")
    print(f"Output directory: {output_dir}")
    print("Mode: WITHOUT geofence filtering")
    
    # === STEP 1: LOAD METADATA ===
    print("Step 1: Loading JSON metadata...")
    if not os.path.exists(json_metadata_path):
        raise FileNotFoundError(f"JSON metadata file not found: {json_metadata_path}")
    
    with open(json_metadata_path, "r") as f:
        metadata = json.load(f)
    
    # === STEP 2: RUN PREDICTIONS WITHOUT GEOFENCE ===
    print("Step 2: Running tree detection WITHOUT geofence filtering...")
    
    # Create a large bounding box that encompasses the entire image (no filtering)
    # This effectively disables geofence filtering by using the image bounds
    
    # Extract bounds from metadata - handle multiple formats
    bounds = None
    if "bounds" in metadata:
        # Original format: bounds at root level
        bounds = metadata["bounds"]
    elif "geographic_info" in metadata and "bounds" in metadata["geographic_info"]:
        # Tile format: bounds nested under geographic_info
        bounds = metadata["geographic_info"]["bounds"]
    else:
        raise ValueError(f"Metadata JSON missing 'bounds' field in expected locations: {json_metadata_path}")
    
    # Use image bounds as a "geofence" that includes everything
    image_bounds = [
        (bounds["min_lon"], bounds["min_lat"]),  # bottom-left
        (bounds["max_lon"], bounds["min_lat"]),  # bottom-right
        (bounds["max_lon"], bounds["max_lat"]),  # top-right
        (bounds["min_lon"], bounds["max_lat"]),  # top-left
        (bounds["min_lon"], bounds["min_lat"])   # close polygon
    ]
    
    predictions, output_img, metrics = predict_from_jpg(
        jpg_path=jpg_path,
        model_path=model_path,
        output_image=output_image,
        output_pred_csv=output_pred_csv,
        output_kml=output_kml,
        json_metadata_path=json_metadata_path,
        geofence_coords=image_bounds,  # Use image bounds as "geofence"
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        iou_threshold=iou_threshold,
        conf_min=conf_min
    )

    # === STEP 3: GENERATE CONFIDENCE HISTOGRAM ===
    print("Step 3: Generating confidence histogram...")
    histogram_path = os.path.join(output_dir, f"{image_name}_confidence_histogram.png")
    generate_confidence_histogram(predictions, histogram_path, location, metrics)
    
    # === COMPILE RESULTS ===
    results = {
        'location': location,
        'processing_mode': 'without_geofence',
        'input_files': {
            'image': jpg_path,
            'metadata': json_metadata_path,
            'model': model_path
        },
        'output_files': {
            'predictions_image': output_image,
            'predictions_csv': output_pred_csv,
            'predictions_kml': output_kml,
            'confidence_histogram': histogram_path
        },
        'metadata': metadata,
        'predictions': predictions,
        'metrics': metrics,
        'geofence_disabled': True
    }
    
    return results


def tile_image_and_labels(location, tile_size=800, overlap_ratio=0.25):
    """
    Tile an image and split corresponding labels into tiles.
    Similar to tile_and_label_split.py functionality.
    
    Args:
        location (str): Location name
        tile_size (int): Size of tiles in pixels (default: 400)
        overlap_ratio (float): Overlap ratio between tiles (default: 0.25)
    
    Returns:
        dict: Summary of tiling process
    """
    print(f"Starting tiling process for {location}")
    
    # === PATHS ===
    image_path = f"data/images/{location}/{location}.jpg"
    metadata_path = f"data/metadata/{location}/{location}_metadata.json"
    label_csv_path = f"data/labels/{location}/{location}_labels.csv"
    
    tile_output_dir = f"data/images/{location}/tiles"
    label_output_dir = f"data/labels/{location}/tiles"
    metadata_output_dir = f"data/metadata/{location}/tiles"
    
    # Create directories
    os.makedirs(tile_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(metadata_output_dir, exist_ok=True)
    
    # === LOAD IMAGE AND METADATA ===
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not os.path.exists(label_csv_path):
        raise FileNotFoundError(f"Labels not found: {label_csv_path}")
    
    img = Image.open(image_path)
    width, height = img.size
    print(f"Image size: {width}x{height}")
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    
    df = pd.read_csv(label_csv_path)
    print(f"Loaded {len(df)} labels")
    
    # === TILE SETTINGS ===
    step = int(tile_size * (1 - overlap_ratio))
    print(f"Tile settings: {tile_size}x{tile_size}px, {overlap_ratio:.0%} overlap, step={step}px")
    
    lon_per_px = (meta["bounds"]["max_lon"] - meta["bounds"]["min_lon"]) / meta["width_px"]
    lat_per_px = (meta["bounds"]["max_lat"] - meta["bounds"]["min_lat"]) / meta["height_px"]
    
    # === TILE AND LABEL ===
    tile_id = 0
    tiles_with_labels = 0
    total_labels_in_tiles = 0
    
    for y_start in range(0, height - tile_size + 1, step):
        for x_start in range(0, width - tile_size + 1, step):
            x_end = x_start + tile_size
            y_end = y_start + tile_size
            
            # Extract tile image
            tile_img = img.crop((x_start, y_start, x_end, y_end))
            tile_name = f"{location}_tile_{tile_id:04d}.jpg"
            tile_img.save(os.path.join(tile_output_dir, tile_name))
            
            # Extract labels inside this tile
            boxes = df[
                (df["xmin"] >= x_start) & (df["xmax"] <= x_end) &
                (df["ymin"] >= y_start) & (df["ymax"] <= y_end)
            ].copy()
            
            if not boxes.empty:
                # Adjust coordinates relative to tile
                boxes["xmin"] -= x_start
                boxes["xmax"] -= x_start
                boxes["ymin"] -= y_start
                boxes["ymax"] -= y_start
                boxes["image_path"] = tile_name
                
                # Save tile labels
                boxes.to_csv(os.path.join(label_output_dir, tile_name.replace(".jpg", ".csv")), index=False)
                tiles_with_labels += 1
                total_labels_in_tiles += len(boxes)
            
            # Save tile metadata
            tile_meta = {
                "file": tile_name,
                "resolution_m_per_pixel": meta["resolution_m_per_pixel"],
                "spatial_reference": meta["spatial_reference"],
                "width_px": tile_size,
                "height_px": tile_size,
                "bounds": {
                    "min_lat": meta["bounds"]["max_lat"] - y_end * lat_per_px,
                    "max_lat": meta["bounds"]["max_lat"] - y_start * lat_per_px,
                    "min_lon": meta["bounds"]["min_lon"] + x_start * lon_per_px,
                    "max_lon": meta["bounds"]["min_lon"] + x_end * lon_per_px,
                }
            }
            
            with open(os.path.join(metadata_output_dir, tile_name.replace(".jpg", "_metadata.json")), "w") as f:
                json.dump(tile_meta, f, indent=2)
            
            tile_id += 1
    
    print(f"Created {tile_id} tiles ({tiles_with_labels} with labels)")
    
    # === COMBINE ALL TILE LABELS INTO ONE CSV ===
    combined_csv_path = f"data/labels/{location}/tiles_labels.csv"
    all_boxes = []
    
    for fname in os.listdir(label_output_dir):
        if fname.endswith(".csv"):
            tile_df = pd.read_csv(os.path.join(label_output_dir, fname))
            all_boxes.append(tile_df)
    
    if all_boxes:
        final_df = pd.concat(all_boxes, ignore_index=True)
        final_df.to_csv(combined_csv_path, index=False)
        print(f"Combined label CSV written to: {combined_csv_path} ({len(final_df)} total labels)")
    else:
        print("WARNING: No tile labels found to combine.")
    
    # === SUMMARY ===
    summary = {
        'location': location,
        'original_image_size': (width, height),
        'tile_size': tile_size,
        'overlap_ratio': overlap_ratio,
        'total_tiles': tile_id,
        'tiles_with_labels': tiles_with_labels,
        'original_labels': len(df),
        'labels_in_tiles': total_labels_in_tiles,
        'combined_csv': combined_csv_path,
        'tile_output_dir': tile_output_dir
    }
    
    return summary


def process_image(jpg_path, json_metadata_path, kml_path, model_path, output_dir, 
                  patch_size=400, patch_overlap=0.25, iou_threshold=0.25, conf_min=0.25):
    """
    Wrapper function that calls the appropriate processing function based on whether KML path is provided.
    
    Args:
        jpg_path (str): Path to input image
        json_metadata_path (str): Path to JSON metadata file
        kml_path (str): Path to KML geofence file (if None, skips geofence filtering)
        model_path (str): Path to trained model checkpoint or "1.5.0" for release model
        output_dir (str): Directory to save all outputs
        patch_size (int): Size of patches for tiled prediction (default: 400)
        patch_overlap (float): Overlap between patches 0.0-1.0 (default: 0.25)
        iou_threshold (float): IoU threshold for NMS (default: 0.25)
        conf_min (float): Minimum confidence threshold (default: 0.25)
    
    Returns:
        dict: Complete results including paths and metrics
    """
    
    if kml_path and os.path.exists(kml_path):
        print("Using geofence filtering (KML file provided)")
        return process_image_with_geofence(
            jpg_path=jpg_path,
            json_metadata_path=json_metadata_path,
            kml_path=kml_path,
            model_path=model_path,
            output_dir=output_dir,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            iou_threshold=iou_threshold,
            conf_min=conf_min
        )
    else:
        if kml_path:
            print(f"Warning: KML file not found at {kml_path}, proceeding without geofence")
        else:
            print("No KML file specified, proceeding without geofence")
        
        return process_image_without_geofence(
            jpg_path=jpg_path,
            json_metadata_path=json_metadata_path,
            model_path=model_path,
            output_dir=output_dir,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            iou_threshold=iou_threshold,
            conf_min=conf_min
        )


# Example usage
if __name__ == "__main__":
    # === CONFIGURATION ===
    location = "Rockfield"  # Change this to your location
    
    # Input files
    jpg_path = fr"C:\Users\andre\Desktop\ArborNote\PalmsID\version03\interface\temp_files\ebd0847b-d845-4216-a26b-3c0a58a33fa0\processed_image.jpg"
    #jpg_path = "data/images/NewportBlvd/tiles/NewportBlvd_tile_00_01.jpg"
    json_metadata_path = fr"C:\Users\andre\Desktop\ArborNote\PalmsID\version03\interface\temp_files\ebd0847b-d845-4216-a26b-3c0a58a33fa0\job_ebd0847b_highres_metadata.json"
    #json_metadata_path = "data/metadata/NewportBlvd/tiles/NewportBlvd_tile_00_01.json"
    kml_path = fr"C:\Users\andre\Desktop\ArborNote\PalmsID\version03\interface\temp_files\ebd0847b-d845-4216-a26b-3c0a58a33fa0\geofence_boundary.kml"
    
    # Model and output
    model_path = fr"C:\Users\andre\Desktop\ArborNote\PalmsID\version03\interface\checkpoints\model_epoch_55.ckpt"  # or "1.5.0" for release
    #model_path = "models/model_epoch_25.ckpt"
    #output_dir = f"data/results/{location}/{location}_tile_00_01"
    output_dir = f"results/"
    
    # Parameters
    conf_min = 0.15
    patch_size = 800
    patch_overlap = 0.25
    iou_threshold = 0.5
    
    print("TREE DETECTION PIPELINE")
    print("=" * 50)
    print(f"Location: {location}")
    print(f"Image: {jpg_path}")
    print(f"Metadata: {json_metadata_path}")
    print(f"Geofence: {kml_path}")
    print(f"Model: {Path(model_path).stem}")
    print(f"Output: {output_dir}")
    
    try:
        # === RUN COMPLETE WORKFLOW ===
        #results = process_image_with_geofence(
        results = process_image_without_geofence(
            jpg_path=jpg_path,
            json_metadata_path=json_metadata_path,
            model_path=model_path,
            output_dir=output_dir,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            iou_threshold=iou_threshold,
            conf_min=conf_min
        )
        
        # === PRINT FINAL SUMMARY ===
        print(f"\nPROCESSING COMPLETE!")
        print(f"   Location: {results['location']}")
        print(f"   Image size: {results['metrics']['image_size']}")
        print(f"   Raw predictions: {results['metrics']['total_raw_predictions']}")
        print(f"   After aspect ratio filter: {results['metrics']['after_aspect_ratio_filter']}")
        print(f"   Aspect ratio filtered: {results['metrics']['aspect_ratio_filtered']}")
        print(f"   Confidence filtered: {results['metrics']['confidence_filtered']}")
        print(f"   Invalid coordinates: {results['metrics']['invalid_coordinates']}")
        print(f"   Geofence filtered: {results['metrics']['geofence_filtered']}")
        print(f"   Final predictions: {results['metrics']['final_predictions']}")
        print(f"   Average confidence: {results['metrics']['avg_confidence']:.3f}")
        print(f"   Confidence threshold: {results['metrics']['confidence_threshold']}")
        
        print(f"\nOUTPUT FILES:")
        print(f"   * Predictions Image: {results['output_files']['predictions_image']}")
        print(f"   * Predictions CSV: {results['output_files']['predictions_csv']}")
        print(f"   * Predictions KML: {results['output_files']['predictions_kml']}")
        print(f"   * Confidence Histogram: {results['output_files']['confidence_histogram']}")
        
        print(f"\nAll files saved successfully!")
        
    except Exception as e:
        print(f"\nERROR: Processing failed: {e}")
        raise
