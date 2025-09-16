import requests
import json
import os
import math
import xml.etree.ElementTree as ET
from PIL import Image
from datetime import datetime

API_KEY = "N2UwODlmZjctZGY1OS00ODVmLWFhZjEtZTlhY2VmZmE3NzJm"  # Working API key

# Current location to extract (change this to extract different locations)
CURRENT_LOCATION = "CabrilloTech"

# 🍂 SEASONAL IMAGE PREFERENCE SETTINGS 🍂
PRIORITIZE_FALL_IMAGERY = True        # Set to True to prefer fall imagery
PREFERRED_MONTHS = [8, 9, 10]         # August=8, September=9, October=10
PREFERRED_YEAR = 2025               # Prioritize 2024 fall imagery over recent
FALLBACK_TO_RECENT = True             # If no preferred season/year found, use most recent
MAX_SURVEYS_TO_CHECK = 50             # How many surveys to check for options

print("🍂 FALL IMAGERY PRIORITIZATION ENABLED" if PRIORITIZE_FALL_IMAGERY else "📅 USING MOST RECENT IMAGERY")
if PRIORITIZE_FALL_IMAGERY:
    month_names = {8: "August", 9: "September", 10: "October", 11: "November"}
    preferred_names = [month_names.get(m, f"Month {m}") for m in PREFERRED_MONTHS]
    print(f"🎯 Preferred: {PREFERRED_YEAR} {', '.join(preferred_names)} > Recent imagery")

def find_available_locations():
    """Find all available locations by scanning for KML files in data folders"""
    data_path = "data"
    locations = []
    
    if os.path.exists(data_path):
        for item in os.listdir(data_path):
            location_path = os.path.join(data_path, item)
            if os.path.isdir(location_path):
                # Look for KML files in the location folder
                kml_files = []
                for file in os.listdir(location_path):
                    if file.endswith('.kml'):
                        kml_files.append(file)
                
                if kml_files:
                    locations.append({
                        "name": item,
                        "kml_files": kml_files,
                        "path": location_path
                    })
    
    return locations

def extract_bbox_from_kml(kml_file_path):
    """Extract bounding box coordinates from KML file"""
    try:
        tree = ET.parse(kml_file_path)
        root = tree.getroot()
        
        # Handle KML namespace
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Find coordinates in Polygon or LinearRing
        coordinates_elem = root.find('.//kml:coordinates', namespace)
        if coordinates_elem is None:
            # Try without namespace
            coordinates_elem = root.find('.//coordinates')
        
        if coordinates_elem is not None:
            coords_text = coordinates_elem.text.strip()
            
            # Parse coordinates (format: "lon,lat,alt lon,lat,alt ...")
            coord_pairs = coords_text.split()
            
            lons = []
            lats = []
            
            for coord_pair in coord_pairs:
                parts = coord_pair.split(',')
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        lons.append(lon)
                        lats.append(lat)
                    except ValueError:
                        continue
            
            if lons and lats:
                min_lon = min(lons)
                max_lon = max(lons)
                min_lat = min(lats)
                max_lat = max(lats)
                
                # Return in bbox format: "min_lon,min_lat,max_lon,max_lat"
                bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
                return bbox
        
        print(f"⚠️ Could not find coordinates in KML file")
        return None
        
    except Exception as e:
        print(f"❌ Error parsing KML file: {e}")
        return None

def get_season_from_date(date_str):
    """Get season name from date string"""
    try:
        from datetime import datetime
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month = date_obj.month
        
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer" 
        elif month in [9, 10, 11]:
            return "Fall"
        else:
            return "Unknown"
    except:
        return "Unknown"

def select_preferred_survey(surveys):
    """Select the best survey based on seasonal preferences"""
    from datetime import datetime
    
    if not surveys:
        return None
    
    print(f"📊 Found {len(surveys)} available surveys:")
    
    # Parse and categorize surveys
    preferred_fall_surveys = []  # Preferred year + fall months
    other_fall_surveys = []      # Other years + fall months
    recent_surveys = []          # All other surveys (sorted by recency)
    
    for survey in surveys:
        capture_date = survey.get('captureDate', '')
        season = get_season_from_date(capture_date)
        
        try:
            date_obj = datetime.strptime(capture_date, "%Y-%m-%d")
            month = date_obj.month
            year = date_obj.year
            
            print(f"   📅 {capture_date} ({season}) - Year {year}")
            
            if PRIORITIZE_FALL_IMAGERY and month in PREFERRED_MONTHS:
                if year == PREFERRED_YEAR:
                    preferred_fall_surveys.append((survey, date_obj))
                    print(f"      🎯 PREFERRED: {PREFERRED_YEAR} Fall imagery")
                else:
                    other_fall_surveys.append((survey, date_obj))
                    print(f"      🍂 Fall imagery (different year)")
            else:
                recent_surveys.append((survey, date_obj))
        except:
            # If date parsing fails, add to recent_surveys with current date
            recent_surveys.append((survey, datetime.now()))
            print(f"   📅 {capture_date} (Unknown date format)")
    
    print(f"\n📈 Survey categorization:")
    print(f"   🎯 Preferred {PREFERRED_YEAR} Fall: {len(preferred_fall_surveys)}")
    print(f"   🍂 Other Fall imagery: {len(other_fall_surveys)}")
    print(f"   📅 Recent/Other seasons: {len(recent_surveys)}")
    
    # Selection logic - Priority order
    if PRIORITIZE_FALL_IMAGERY and preferred_fall_surveys:
        # 1st Priority: Preferred year + fall months (most recent within that category)
        preferred_fall_surveys.sort(key=lambda x: x[1], reverse=True)
        selected_survey = preferred_fall_surveys[0][0]
        selected_date = preferred_fall_surveys[0][1]
        
        print(f"\n🎯 SELECTED: {PREFERRED_YEAR} Fall imagery: {selected_survey['captureDate']}")
        print(f"   Priority Level: 1 (Preferred year + season)")
        return selected_survey
        
    elif PRIORITIZE_FALL_IMAGERY and other_fall_surveys:
        # 2nd Priority: Other years + fall months (most recent fall imagery)
        other_fall_surveys.sort(key=lambda x: x[1], reverse=True)
        selected_survey = other_fall_surveys[0][0]
        selected_date = other_fall_surveys[0][1]
        
        print(f"\n🍂 SELECTED: Fall imagery from different year: {selected_survey['captureDate']}")
        print(f"   Priority Level: 2 (Fall season, different year)")
        return selected_survey
        
    elif FALLBACK_TO_RECENT and recent_surveys:
        # 3rd Priority: Most recent imagery (any season/year)
        recent_surveys.sort(key=lambda x: x[1], reverse=True)
        selected_survey = recent_surveys[0][0]
        
        print(f"\n📅 SELECTED: Most recent imagery: {selected_survey['captureDate']}")
        print(f"   Priority Level: 3 (Fallback to most recent)")
        if PRIORITIZE_FALL_IMAGERY:
            print(f"   ⚠️ No fall imagery available from any year")
        
        return selected_survey
    
    else:
        # Last resort: First available survey
        if surveys:
            selected_survey = surveys[0]
            print(f"\n⚠️ FALLBACK: Using first available survey: {selected_survey['captureDate']}")
            print(f"   Priority Level: 4 (Emergency fallback)")
            return selected_survey
    
    print(f"\n❌ No suitable surveys found")
    return None

def get_location_bbox(location_name):
    """Get bbox for a location by reading its KML file"""
    data_path = f"data/{location_name}"
    
    if not os.path.exists(data_path):
        print(f"❌ Location folder not found: {data_path}")
        return None
    
    # Look for KML files
    kml_files = [f for f in os.listdir(data_path) if f.endswith('.kml')]
    
    if not kml_files:
        print(f"❌ No KML files found in {data_path}")
        return None
    
    # Use the first KML file (or prioritize the one named after the location)
    primary_kml = None
    for kml_file in kml_files:
        if location_name.lower() in kml_file.lower():
            primary_kml = kml_file
            break
    
    if not primary_kml:
        primary_kml = kml_files[0]
    
    kml_path = os.path.join(data_path, primary_kml)
    print(f"📍 Reading bbox from: {kml_path}")
    
    bbox = extract_bbox_from_kml(kml_path)
    return bbox

# Get bbox from KML file
bbox = get_location_bbox(CURRENT_LOCATION)
location_name = CURRENT_LOCATION

if bbox is None:
    print(f"❌ Could not extract bbox for {CURRENT_LOCATION}")
    print(f"Available locations:")
    locations = find_available_locations()
    for loc in locations:
        print(f"  📁 {loc['name']} (KML files: {', '.join(loc['kml_files'])})")
    exit(1)

print("🗺️ NEARMAP EXTRACTION - MATCHING YOUR DATA STRUCTURE")
print("=" * 60)
print(f"📦 Location: {location_name}")
print(f"📦 Bounding Box: {bbox}")
print(f"🎯 Creating folder structure like your existing data/")

def crop_black_borders(image):
    """Remove black borders from the combined image"""
    import numpy as np
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Find non-black pixels (pixels that aren't pure black [0,0,0])
    # Allow for slight variations (threshold of 10 to account for compression artifacts)
    non_black_mask = np.any(img_array > 10, axis=2)
    
    # Find the bounding box of non-black content
    rows = np.any(non_black_mask, axis=1)
    cols = np.any(non_black_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        print("⚠️ Warning: No non-black content found, keeping original image")
        return image, image.size
    
    # Get the crop boundaries
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    # Add small padding to avoid cutting off edges
    padding = 10
    height, width = img_array.shape[:2]
    
    top = max(0, top - padding)
    bottom = min(height - 1, bottom + padding)
    left = max(0, left - padding)
    right = min(width - 1, right + padding)
    
    # Crop the image
    cropped_image = image.crop((left, top, right + 1, bottom + 1))
    original_size = image.size
    new_size = cropped_image.size
    
    print(f"🔄 Cropped image from {original_size[0]}x{original_size[1]} to {new_size[0]}x{new_size[1]}")
    print(f"   Removed: {original_size[0] - new_size[0]} width, {original_size[1] - new_size[1]} height in black borders")
    
    return cropped_image, new_size

def create_location_folders(location_name):
    """Create the exact folder structure matching your existing data"""
    base_path = f"data/{location_name}"
    folders = ["images", "labels", "metadata"]
    
    created_folders = {}
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        created_folders[folder] = os.path.abspath(folder_path)
        print(f"📁 Created/verified: {created_folders[folder]}")
    
    return created_folders

def create_highres_version(location_name, folders, original_image_path, target_resolution=0.075):
    """Create a high-resolution version if current resolution is lower than target"""
    
    print(f"\n🔍 Step 6: Checking if high-resolution version is needed...")
    
    # Load the metadata to get current resolution
    metadata_file = os.path.join(folders["metadata"], f"{location_name}_metadata.json")
    
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    current_resolution = metadata.get("resolution_m_per_pixel", 0)
    
    print(f"   Current resolution: {current_resolution:.6f} m/px")
    print(f"   Target resolution: {target_resolution:.6f} m/px")
    
    # Check if we need to create high-res version
    if current_resolution >= target_resolution:
        print(f"   ✅ Resolution is already {target_resolution} m/px or higher - no high-res version needed")
        return False
    
    print(f"   🎯 Creating high-resolution version at {target_resolution} m/px...")
    
    # Calculate scale factor
    scale_factor = current_resolution / target_resolution
    print(f"   Scale factor: {scale_factor:.4f}")
    
    # Load original image
    from PIL import Image
    original_image = Image.open(original_image_path).convert("RGB")
    print(f"   Original image size: {original_image.width}×{original_image.height} pixels")
    
    # Calculate new dimensions
    new_width = int(round(original_image.width * scale_factor))
    new_height = int(round(original_image.height * scale_factor))
    
    print(f"   Resizing: {original_image.width}×{original_image.height} → {new_width}×{new_height}")
    
    # Resize image using high-quality resampling
    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Save high-res version
    highres_filename = f"{location_name}_highres.jpg"
    highres_path = os.path.join(folders["images"], highres_filename)
    resized_image.save(highres_path, 'JPEG', quality=95)
    
    print(f"   ✅ High-res image saved: {highres_path}")
    print(f"   📐 Final dimensions: {new_width}×{new_height}")
    
    # Create high-res metadata
    highres_metadata = metadata.copy()
    highres_metadata.update({
        "file": highres_filename,
        "resolution_m_per_pixel": round(target_resolution, 3),
        "width_px": new_width,
        "height_px": new_height,
        "original_image_size": {
            "width": metadata["width_px"],
            "height": metadata["height_px"]
        },
        "original_resolution_meters_per_pixel": current_resolution,
        "scale_factor_applied": scale_factor,
        "processing_note": f"Resized from {current_resolution:.6f} m/px to {target_resolution:.6f} m/px for high-resolution training",
        "created_from": metadata["file"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Save high-res metadata
    highres_metadata_file = os.path.join(folders["metadata"], f"{location_name}_highres_metadata.json")
    with open(highres_metadata_file, 'w') as f:
        json.dump(highres_metadata, f, indent=4)
    
    print(f"   💾 High-res metadata saved: {highres_metadata_file}")
    
    # Clean up memory
    original_image.close()
    resized_image.close()
    
    # Calculate size reduction percentage
    original_pixels = metadata["width_px"] * metadata["height_px"]
    new_pixels = new_width * new_height
    size_change = ((new_pixels - original_pixels) / original_pixels) * 100
    
    print(f"   📊 Size change: {size_change:+.1f}% ({original_pixels:,} → {new_pixels:,} pixels)")
    
    return True

def save_metadata_matching_format(location_name, bbox, survey_data, image_info, folders):
    """Save metadata in the EXACT format of your existing CabrilloTech_metadata.json"""
    
    # Parse bbox coordinates
    bbox_coords = bbox.split(',')
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_coords)
    
    # Calculate resolution per pixel (matching your 0.075 format)
    lat_rad = math.radians((min_lat + max_lat) / 2)
    lon_distance_m = abs(max_lon - min_lon) * 111320 * math.cos(lat_rad)
    resolution_m_per_pixel = lon_distance_m / image_info["width"]
    
    # Create metadata in YOUR EXACT FORMAT with extraction date
    metadata = {
        "file": image_info["filename"],
        "resolution_m_per_pixel": round(resolution_m_per_pixel, 3),
        "spatial_reference": "EPSG:3857",
        "width_px": image_info["width"],
        "height_px": image_info["height"],
        "bounds": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        },
        "extraction_info": {
            "extracted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "survey_capture_date": survey_data.get("captureDate", "Unknown"),
            "survey_id": survey_data.get("id", "Unknown"),
            "nearmap_api_version": "v3",
            "bbox_used": bbox,
            "seasonal_selection": {
                "prioritize_fall": PRIORITIZE_FALL_IMAGERY,
                "preferred_months": PREFERRED_MONTHS if PRIORITIZE_FALL_IMAGERY else [],
                "preferred_year": PREFERRED_YEAR if PRIORITIZE_FALL_IMAGERY else None,
                "selected_season": get_season_from_date(survey_data.get("captureDate", "")),
                "total_surveys_available": survey_data.get("total_surveys_checked", "Unknown")
            }
        }
    }
    
    # Save in the exact same format as your existing metadata
    metadata_file = os.path.join(folders["metadata"], f"{location_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"💾 Metadata saved (matching your format): {metadata_file}")
    return metadata_file

def extract_nearmap_with_your_structure(location_name, bbox):
    """Extract nearmap image using the tile approach that created your CabrilloTech image"""
    
    # Step 1: Create matching folder structure
    print(f"\n📁 Step 1: Creating folder structure like your existing data...")
    folders = create_location_folders(location_name)
    
    # Step 2: Get coverage data using working endpoint
    print(f"\n🔍 Step 2: Getting coverage data...")
    coverage_url = f"https://api.nearmap.com/coverage/v2/tx/bbox/{bbox}"
    coverage_params = {
        "dates": "all",
        "resources": "tiles:Vert", 
        "limit": str(MAX_SURVEYS_TO_CHECK),  # Get more options to choose from
        "apikey": API_KEY
    }
    
    print(f"URL: {coverage_url}")
    if PRIORITIZE_FALL_IMAGERY:
        print(f"🍂 Looking for {PREFERRED_YEAR} fall imagery (Aug/Sep/Oct) first...")
    
    try:
        cov_response = requests.get(coverage_url, params=coverage_params, timeout=15)
        print(f"📊 Coverage Status: {cov_response.status_code}")
        
        if cov_response.status_code == 200:
            cov_data = cov_response.json()
            surveys = cov_data.get('surveys', [])
            transaction_token = cov_data.get('transactionToken', '')
            
            if surveys and transaction_token:
                # NEW: Select survey based on seasonal preference
                selected_survey = select_preferred_survey(surveys)
                
                if selected_survey:
                    survey = selected_survey
                    survey_id = survey['id']
                    capture_date = survey['captureDate']
                    scale_info = survey.get('scale', {}).get('raster:Vert', {})
                    
                    width_tiles = scale_info.get('widthInTiles', 1)
                    height_tiles = scale_info.get('heightInTiles', 1)
                    
                    print(f"\n✅ Selected survey details:")
                    print(f"   🆔 Survey ID: {survey_id}")
                    print(f"   📅 Capture Date: {capture_date}")
                    print(f"   🍂 Season: {get_season_from_date(capture_date)}")
                    print(f"   📏 Grid Size: {width_tiles}x{height_tiles} tiles")
                    print(f"   🔑 Transaction Token: {transaction_token[:50]}...")
                else:
                    print(f"❌ No suitable survey found")
                    return False
                
                # Step 3: Download tiles (Nearmap delivers high-res imagery in tiles)
                print(f"\n🧩 Step 3: Downloading {width_tiles * height_tiles} tiles...")
                print(f"   ℹ️ Why tiles? Nearmap uses tiled delivery for high-resolution imagery.")
                print(f"   Each tile is 4096x4096 pixels for maximum detail and faster download.")
                
                tile_url = f"https://api.nearmap.com/staticmap/v3/surveys/{survey_id}/Vert.jpg"
                tiles = []
                total_tiles = width_tiles * height_tiles
                
                for y in range(height_tiles):
                    for x in range(width_tiles):
                        tile_num = y * width_tiles + x + 1
                        print(f"📥 Downloading tile {tile_num}/{total_tiles} ({x},{y})...")
                        
                        tile_params = {
                            "x": str(x),
                            "y": str(y),
                            "tileSize": "4096x4096",
                            "transactionToken": transaction_token,
                            "apikey": API_KEY
                        }
                        
                        tile_response = requests.get(tile_url, params=tile_params, stream=True, timeout=30)
                        
                        if tile_response.status_code == 200:
                            tile_filename = f"temp_tile_{x}_{y}.jpg"
                            with open(tile_filename, "wb") as tile_file:
                                for chunk in tile_response.iter_content(1024):
                                    tile_file.write(chunk)
                            
                            print(f"   ✅ Downloaded: {tile_filename}")
                            tiles.append((x, y, tile_filename))
                        else:
                            print(f"   ❌ Failed tile ({x},{y}): {tile_response.status_code}")
                
                # Step 4: Combine tiles and remove black borders
                if tiles:
                    print(f"\n🔧 Step 4: Combining {len(tiles)} tiles and removing black borders...")
                    
                    # Load first tile to get dimensions
                    first_tile = Image.open(tiles[0][2])
                    tile_width, tile_height = first_tile.size
                    first_tile.close()
                    
                    # Create combined image
                    total_width = width_tiles * tile_width
                    total_height = height_tiles * tile_height
                    combined_image = Image.new('RGB', (total_width, total_height))
                    
                    print(f"📐 Tile size: {tile_width}x{tile_height}")
                    print(f"📐 Combined size (before crop): {total_width}x{total_height}")
                    
                    # Paste each tile
                    for x, y, filename in tiles:
                        tile_img = Image.open(filename)
                        paste_x = x * tile_width
                        paste_y = y * tile_height
                        combined_image.paste(tile_img, (paste_x, paste_y))
                        tile_img.close()
                    
                    # Crop black borders (this removes the empty space around your actual imagery)
                    print(f"✂️ Cropping black borders...")
                    cropped_image, final_size = crop_black_borders(combined_image)
                    combined_image.close()  # Free memory
                    
                    # Save in images folder (matching your structure)
                    output_filename = f"{location_name}_nearmap.jpg"
                    output_path = os.path.join(folders["images"], output_filename)
                    cropped_image.save(output_path, 'JPEG', quality=95)
                    
                    print(f"\n🎉 SUCCESS! Image saved: {output_path}")
                    print(f"📐 Final dimensions: {final_size[0]}x{final_size[1]} (black borders removed)")
                    
                    # Step 5: Save metadata in your exact format
                    print(f"\n💾 Step 5: Saving metadata (matching your format)...")
                    image_info = {
                        "filename": output_filename,
                        "width": final_size[0],  # Use cropped dimensions
                        "height": final_size[1],  # Use cropped dimensions
                        "tile_size": f"{tile_width}x{tile_height}",
                        "width_tiles": width_tiles,
                        "height_tiles": height_tiles,
                        "total_tiles": len(tiles),
                        "original_size": f"{total_width}x{total_height}",
                        "cropped_size": f"{final_size[0]}x{final_size[1]}"
                    }
                    
                    # Add survey metadata for tracking
                    enhanced_survey = survey.copy()
                    enhanced_survey["total_surveys_checked"] = len(surveys)
                    
                    metadata_file = save_metadata_matching_format(location_name, bbox, enhanced_survey, image_info, folders)
                    cropped_image.close()  # Free memory
                    
                    # Step 6: Check resolution and create high-res version if needed
                    highres_created = create_highres_version(location_name, folders, output_path, target_resolution=0.075)
                    
                    # Step 7: Clean up temp files
                    print(f"\n🧹 Step 7: Cleaning up temp files...")
                    for x, y, filename in tiles:
                        try:
                            os.remove(filename)
                            print(f"   🗑️ Removed: {filename}")
                        except:
                            pass
                    
                    # Final summary
                    print(f"\n" + "="*60)
                    print(f"✨ EXTRACTION COMPLETE FOR {location_name.upper()}! ✨")
                    print(f"="*60)
                    print(f"📸 Original Image: data/{location_name}/images/{output_filename}")
                    if highres_created:
                        print(f"🎯 High-Res Image: data/{location_name}/images/{location_name}_highres.jpg (0.075 m/px)")
                        print(f"💾 High-Res Metadata: data/{location_name}/metadata/{location_name}_highres_metadata.json")
                    print(f"💾 Original Metadata: data/{location_name}/metadata/{location_name}_metadata.json")
                    print(f"📁 Labels folder ready: data/{location_name}/labels/")
                    print(f"📊 Resolution: ~{round(image_info['width']/1000, 1)}K x {round(image_info['height']/1000, 1)}K pixels")
                    print(f"📅 Survey date: {capture_date}")
                    print(f"="*60)
                    
                    return True
                    
                else:
                    print(f"❌ No tiles downloaded successfully")
                    return False
            else:
                print(f"❌ No surveys or transaction token in response")
                print(f"Raw response: {json.dumps(cov_data, indent=2)[:500]}...")
                return False
        else:
            print(f"❌ Coverage request failed: {cov_response.status_code}")
            print(f"Response: {cov_response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main execution
if __name__ == "__main__":
    print(f"\n🚀 Starting extraction for {location_name}...")
    print(f"🎯 This will create the same structure as your existing data folders")
    print(f"📦 Bbox extracted from KML: {bbox}")
    
    success = extract_nearmap_with_your_structure(location_name, bbox)
    
    if success:
        print(f"\n🎊 MISSION ACCOMPLISHED! 🎊")
        print(f"\n✅ Your data is now organized exactly like your existing structure:")
        print(f"   📸 data/{location_name}/images/ - High-resolution satellite imagery")
        print(f"   � Automatic high-res version created (0.075 m/px) if needed")
        print(f"   �🏷️ data/{location_name}/labels/ - Ready for your CSV annotation files")
        print(f"   💾 data/{location_name}/metadata/ - JSON metadata (same format as CabrilloTech)")
        
        print(f"\n🔄 The folder structure matches your existing data perfectly!")
        
    else:
        print(f"\n❌ Extraction failed. Check the error messages above.")
        
    # Show all available locations from data folders
    print(f"\n📍 ALL YOUR DATA LOCATIONS (with KML files):")
    print(f"-" * 60)
    available_locations = find_available_locations()
    
    for location in available_locations:
        loc_name = location["name"]
        kml_files = location["kml_files"]
        
        if loc_name == CURRENT_LOCATION:
            status = "✅ EXTRACTED"
        elif os.path.exists(f"data/{loc_name}/images"):
            status = "� HAS IMAGES"
        else:
            status = "⏸️ READY TO EXTRACT"
        
        print(f"  {status} {loc_name}")
        print(f"      📄 KML files: {', '.join(kml_files)}")
        
        # Try to extract bbox to show it
        try:
            test_bbox = get_location_bbox(loc_name)
            if test_bbox:
                print(f"      📦 Bbox: {test_bbox}")
        except:
            print(f"      ❌ Could not read bbox from KML")
    
    if not available_locations:
        print(f"  ❌ No locations with KML files found in data/")
    
    print(f"\n💡 TO EXTRACT A DIFFERENT LOCATION:")
    print(f"   1. Change line 12: CURRENT_LOCATION = 'LocationName'")
    print(f"   2. Make sure the location has a KML file in data/LocationName/")
    print(f"   3. Run the script again")
    print(f"   4. The bbox will be automatically extracted from the KML!")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Verify the extracted image quality")
    print(f"   2. Add labels to data/{location_name}/labels/ (CSV format)")
    print(f"   3. Use the metadata for technical specifications")
    print(f"   4. All folders now match your existing data organization!")
    
    print(f"\n✨ NEW: AUTOMATIC HIGH-RESOLUTION VERSION:")
    print(f"   🎯 Automatically creates {location_name}_highres.jpg at 0.075 m/px if needed")
    print(f"   📊 Perfect for machine learning training at consistent resolution")
    print(f"   💾 Includes matching metadata with processing details")
    print(f"   🔄 Uses high-quality LANCZOS resampling for best results")
    
    print(f"\n🍂 NEW: SEASONAL IMAGERY PRIORITIZATION:")
    if PRIORITIZE_FALL_IMAGERY:
        month_names = {8: "August", 9: "September", 10: "October", 11: "November"}
        preferred_names = [month_names.get(m, f"Month {m}") for m in PREFERRED_MONTHS]
        print(f"   🎯 Prioritizes {PREFERRED_YEAR} {', '.join(preferred_names)} over recent imagery")
        print(f"   🍂 Falls back to other years' fall imagery if {PREFERRED_YEAR} unavailable")
        print(f"   📅 Uses most recent imagery only if no fall imagery found")
        print(f"   📊 Checks up to {MAX_SURVEYS_TO_CHECK} available surveys for best match")
    else:
        print(f"   📅 Using most recent imagery (seasonal prioritization disabled)")
    
    print(f"\n✨ BENEFITS OF KML-BASED BBOX EXTRACTION:")
    print(f"   📍 No hardcoded coordinates - reads directly from KML")
    print(f"   🎯 Always accurate boundaries for each location")
    print(f"   🔄 Automatically detects all available locations")
    print(f"   📄 Works with any KML file in your data folders")
    print(f"   🎚️ Smart resolution optimization for consistent training data")
    print(f"   🍂 Intelligent seasonal imagery selection for consistent datasets")
