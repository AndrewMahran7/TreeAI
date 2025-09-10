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

def find_available_locations():
    """Find all available locations by scanning for KML files in data folders"""
    data_path = "../../data"
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

def get_location_bbox(location_name):
    """Get bbox for a location by reading its KML file"""
    data_path = f"../../data/{location_name}"
    
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

def create_location_folders(location_name):
    """Create the exact folder structure matching your existing data"""
    base_path = f"../../data/{location_name}"
    folders = ["images", "labels", "metadata"]
    
    created_folders = {}
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        created_folders[folder] = os.path.abspath(folder_path)
        print(f"📁 Created/verified: {created_folders[folder]}")
    
    return created_folders

def save_metadata_matching_format(location_name, bbox, survey_data, image_info, folders):
    """Save metadata in the EXACT format of your existing CabrilloTech_metadata.json"""
    
    # Parse bbox coordinates
    bbox_coords = bbox.split(',')
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_coords)
    
    # Calculate resolution per pixel (matching your 0.075 format)
    lat_rad = math.radians((min_lat + max_lat) / 2)
    lon_distance_m = abs(max_lon - min_lon) * 111320 * math.cos(lat_rad)
    resolution_m_per_pixel = lon_distance_m / image_info["width"]
    
    # Create metadata in YOUR EXACT FORMAT
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
        "limit": "1",
        "apikey": API_KEY
    }
    
    print(f"URL: {coverage_url}")
    
    try:
        cov_response = requests.get(coverage_url, params=coverage_params, timeout=15)
        print(f"📊 Coverage Status: {cov_response.status_code}")
        
        if cov_response.status_code == 200:
            cov_data = cov_response.json()
            surveys = cov_data.get('surveys', [])
            transaction_token = cov_data.get('transactionToken', '')
            
            if surveys and transaction_token:
                survey = surveys[0]
                survey_id = survey['id']
                capture_date = survey['captureDate']
                scale_info = survey.get('scale', {}).get('raster:Vert', {})
                
                width_tiles = scale_info.get('widthInTiles', 1)
                height_tiles = scale_info.get('heightInTiles', 1)
                
                print(f"✅ Found survey data:")
                print(f"   🆔 Survey ID: {survey_id}")
                print(f"   📅 Capture Date: {capture_date}")
                print(f"   📏 Grid Size: {width_tiles}x{height_tiles} tiles")
                print(f"   🔑 Transaction Token: {transaction_token[:50]}...")
                
                # Step 3: Download tiles (same approach that worked before)
                print(f"\n🧩 Step 3: Downloading {width_tiles * height_tiles} tiles...")
                
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
                
                # Step 4: Combine tiles (same as before)
                if tiles:
                    print(f"\n🔧 Step 4: Combining {len(tiles)} tiles...")
                    
                    # Load first tile to get dimensions
                    first_tile = Image.open(tiles[0][2])
                    tile_width, tile_height = first_tile.size
                    first_tile.close()
                    
                    # Create combined image
                    total_width = width_tiles * tile_width
                    total_height = height_tiles * tile_height
                    combined_image = Image.new('RGB', (total_width, total_height))
                    
                    print(f"📐 Tile size: {tile_width}x{tile_height}")
                    print(f"📐 Final size: {total_width}x{total_height}")
                    
                    # Paste each tile
                    for x, y, filename in tiles:
                        tile_img = Image.open(filename)
                        paste_x = x * tile_width
                        paste_y = y * tile_height
                        combined_image.paste(tile_img, (paste_x, paste_y))
                        tile_img.close()
                    
                    # Save in images folder (matching your structure)
                    output_filename = f"{location_name}_nearmap.jpg"
                    output_path = os.path.join(folders["images"], output_filename)
                    combined_image.save(output_path, 'JPEG', quality=95)
                    combined_image.close()
                    
                    print(f"\n🎉 SUCCESS! Image saved: {output_path}")
                    print(f"📐 Dimensions: {total_width}x{total_height}")
                    
                    # Step 5: Save metadata in your exact format
                    print(f"\n💾 Step 5: Saving metadata (matching your format)...")
                    image_info = {
                        "filename": output_filename,
                        "width": total_width,
                        "height": total_height,
                        "tile_size": f"{tile_width}x{tile_height}",
                        "width_tiles": width_tiles,
                        "height_tiles": height_tiles,
                        "total_tiles": len(tiles)
                    }
                    
                    metadata_file = save_metadata_matching_format(location_name, bbox, survey, image_info, folders)
                    
                    # Step 6: Clean up temp files
                    print(f"\n🧹 Step 6: Cleaning up temp files...")
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
                    print(f"📸 Image: data/{location_name}/images/{output_filename}")
                    print(f"💾 Metadata: data/{location_name}/metadata/{location_name}_metadata.json")
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
        print(f"   🏷️ data/{location_name}/labels/ - Ready for your CSV annotation files")
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
        elif os.path.exists(f"../../data/{loc_name}/images"):
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
        print(f"  ❌ No locations with KML files found in ../../data/")
    
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
    
    print(f"\n✨ BENEFITS OF KML-BASED BBOX EXTRACTION:")
    print(f"   📍 No hardcoded coordinates - reads directly from KML")
    print(f"   🎯 Always accurate boundaries for each location")
    print(f"   🔄 Automatically detects all available locations")
    print(f"   📄 Works with any KML file in your data folders")
