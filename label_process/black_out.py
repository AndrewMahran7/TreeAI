import os
import numpy as np
import cv2
import json
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
from PIL import Image, ImageDraw
from pykml import parser
from shapely.geometry import Point, Polygon
from pathlib import Path


def load_metadata(metadata_path):
    """Load metadata JSON file with image bounds and dimensions."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata from: {metadata_path}")
    return metadata


def load_geofence_from_kml(kml_path):
    """Load geofence coordinates from KML file."""
    with open(kml_path, "r", encoding="utf-8") as file:
        root = parser.parse(file).getroot()

    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    coords_text = None

    polygons = root.xpath(".//kml:Polygon", namespaces=namespace)
    if polygons:
        coords_text = polygons[0].xpath(".//kml:coordinates", namespaces=namespace)[0].text.strip()
        print(f"   Found Polygon with coordinates")

    if not coords_text:
        polygons = root.xpath(".//Polygon")
        if polygons:
            coords_elements = polygons[0].xpath(".//coordinates")
            if coords_elements:
                coords_text = coords_elements[0].text.strip()
                print(f"   Found Polygon without namespace")

    if not coords_text:
        linear_rings = root.xpath(".//kml:LinearRing", namespaces=namespace)
        if linear_rings:
            coords_text = linear_rings[0].xpath(".//kml:coordinates", namespaces=namespace)[0].text.strip()
            print(f"   Found LinearRing with coordinates")

    if not coords_text:
        linear_rings = root.xpath(".//LinearRing")
        if linear_rings:
            coords_elements = linear_rings[0].xpath(".//coordinates")
            if coords_elements:
                coords_text = coords_elements[0].text.strip()
                print(f"   Found LinearRing without namespace")

    if not coords_text:
        coords_elements = root.xpath(".//kml:coordinates", namespaces=namespace)
        if coords_elements:
            coords_text = coords_elements[0].text.strip()
            print(f"   Found coordinates directly")
        else:
            coords_elements = root.xpath(".//coordinates")
            if coords_elements:
                coords_text = coords_elements[0].text.strip()
                print(f"   Found coordinates directly (no namespace)")

    if not coords_text:
        raise ValueError("❌ Could not find coordinates in KML file")

    geofence_coords = []
    for coord_str in coords_text.split():
        coord_str = coord_str.strip()
        if coord_str:
            parts = coord_str.split(",")
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    geofence_coords.append((lon, lat))
                except ValueError:
                    continue

    if not geofence_coords:
        raise ValueError("No valid coordinate pairs found in KML file")

    print(f"✅ Loaded geofence with {len(geofence_coords)} vertices from: {kml_path}")
    return geofence_coords


def get_image_dimensions_from_metadata(metadata):
    if "width_px" in metadata and "height_px" in metadata:
        return metadata["width_px"], metadata["height_px"]
    elif "image_size" in metadata:
        return metadata["image_size"]["width"], metadata["image_size"]["height"]
    else:
        raise KeyError(f"Could not find image dimensions in metadata. Keys: {list(metadata.keys())}")


def latlon_to_pixel(lat, lon, metadata):
    min_lat = metadata["bounds"]["min_lat"]
    max_lat = metadata["bounds"]["max_lat"]
    min_lon = metadata["bounds"]["min_lon"]
    max_lon = metadata["bounds"]["max_lon"]

    width_px, height_px = get_image_dimensions_from_metadata(metadata)

    x = (lon - min_lon) / (max_lon - min_lon) * width_px
    y = height_px - ((lat - min_lat) / (max_lat - min_lat) * height_px)

    return int(x), int(y)


def create_geofence_mask(image_shape, geofence_coords, metadata):
    height, width = image_shape[:2]

    pixel_coords = []
    for lon, lat in geofence_coords:
        x, y = latlon_to_pixel(lat, lon, metadata)
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        pixel_coords.append((x, y))

    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    if len(pixel_coords) >= 3:
        draw.polygon(pixel_coords, fill=255)
    mask = np.array(mask_img)
    return (mask > 128).astype(np.uint8)


def load_labels_csv(labels_path):
    if not os.path.exists(labels_path):
        print(f"⚠️ Labels file not found: {labels_path}")
        return []

    try:
        # Check if file is empty or has no content
        with open(labels_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️ Labels file is empty: {labels_path}")
                return []
        
        df = pd.read_csv(labels_path) if HAS_PANDAS else None
        
        if not HAS_PANDAS:
            # Fallback CSV reading without pandas
            import csv
            ignore_boxes = []
            with open(labels_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('label', '').lower() == 'ignore':
                        bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                        ignore_boxes.append(bbox)
            return ignore_boxes
        
        # Check if DataFrame is empty
        if df.empty:
            print(f"⚠️ Labels file has no data: {labels_path}")
            return []
        
        # Check if required columns exist
        required_columns = ['label', 'xmin', 'ymin', 'xmax', 'ymax']
        if not all(col in df.columns for col in required_columns):
            print(f"⚠️ Labels file missing required columns. Found: {list(df.columns)}")
            return []
        
        ignore_df = df[df['label'].str.lower() == 'ignore']
        ignore_boxes = []
        for _, row in ignore_df.iterrows():
            bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
            ignore_boxes.append(bbox)
        
        if ignore_boxes:
            print(f"✅ Loaded {len(ignore_boxes)} ignore boxes from labels file")
        else:
            print(f"ℹ️ No ignore boxes found in labels file")
            
        return ignore_boxes
        
    except Exception as e:
        if HAS_PANDAS and hasattr(e, '__module__') and 'pandas' in str(e.__module__):
            print(f"⚠️ Labels file is empty or has no columns: {labels_path}")
        else:
            print(f"⚠️ Error loading labels file {labels_path}: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Error loading labels file {labels_path}: {e}")
        return []


def black_out_ignore_boxes(image, ignore_boxes):
    result_image = image.copy()
    for xmin, ymin, xmax, ymax in ignore_boxes:
        result_image[ymin:ymax, xmin:xmax] = 0
    return result_image


def black_out_outside_geofence(image_path, metadata_path, kml_path, output_path, save_mask=False, labels_path=None):
    print(f"BLACKING OUT PIXELS OUTSIDE GEOFENCE AND IGNORE BOXES")
    print("=" * 60)

    # Load image with Pillow to bypass OpenCV limit
    Image.MAX_IMAGE_PIXELS = None
    pil_img = Image.open(image_path)
    image = np.array(pil_img)

    # ✅ Convert RGB → BGR so OpenCV saves colors correctly
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width = image.shape[:2]
    print(f"   Loaded image: {width}x{height} pixels (via Pillow + converted to BGR)")

    metadata = load_metadata(metadata_path)
    geofence_coords = load_geofence_from_kml(kml_path)
    ignore_boxes = load_labels_csv(labels_path) if labels_path else []

    mask = create_geofence_mask(image.shape, geofence_coords, metadata)
    mask_3d = np.stack([mask, mask, mask], axis=2)
    masked_image = image * mask_3d

    if ignore_boxes:
        masked_image = black_out_ignore_boxes(masked_image, ignore_boxes)

    cv2.imwrite(output_path, masked_image)
    print(f"✅ Saved masked image to: {output_path}")

    if save_mask:
        mask_path = output_path.replace('.jpg', '_mask.jpg').replace('.png', '_mask.png')
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        print(f"✅ Saved mask to: {mask_path}")


def main():
    location = "Rockfield"
    black_out_outside_geofence(
        image_path=f"data/{location}/images/{location}_highres.jpg",
        metadata_path=f"data/{location}/metadata/{location}_highres_metadata.json",
        kml_path=f"data/{location}/{location}.kml",
        output_path=f"data/{location}/images/{location}_highres_blacked_out.jpg",
        save_mask=True,
        labels_path=f"data/{location}/labels/{location}_highres_labels.csv"
    )


if __name__ == "__main__":
    main()
