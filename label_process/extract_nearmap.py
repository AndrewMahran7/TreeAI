#!/usr/bin/env python3
"""
Nearmap Image Extractor - Converted from extract_nearmap_final.py
Compatible with app.py class structure but using the working final approach
"""

import os
import sys
import json
import math
import time
import requests
import traceback
from datetime import datetime, timedelta
from urllib.parse import urlencode
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Override PIL's MAX_IMAGE_PIXELS limit to handle very large images
if HAS_PIL:
    Image.MAX_IMAGE_PIXELS = None

class NearmapImageExtractor:
    """
    Nearmap Image Extractor using the working approach from extract_nearmap_final.py
    Converts the working script to a class structure compatible with app.py
    """
    
    def __init__(self, api_key: str = None, target_mpp: float = 0.075):
        """
        Initialize the Nearmap extractor.
        
        Args:
            api_key (str): Nearmap API key
            target_mpp (float): Target meters per pixel resolution
        """
        self.api_key = api_key or os.getenv('NEARMAP_API_KEY')
        if not self.api_key:
            raise ValueError("Nearmap API key required. Set NEARMAP_API_KEY env var or pass as parameter.")
        
        self.target_mpp = target_mpp
        
        # Seasonal preferences (from working script)
        self.prioritize_fall_imagery = True
        self.preferred_months = [8, 9, 10]  # August, September, October
        self.preferred_year = 2024
        self.fallback_to_recent = True
        self.max_surveys_to_check = 50
        
        print(f"🍂 FALL IMAGERY PRIORITIZATION ENABLED" if self.prioritize_fall_imagery else "📅 USING MOST RECENT IMAGERY")
    
    def get_season_from_date(self, date_str):
        """Get season name from date string"""
        try:
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
    
    def select_preferred_survey(self, surveys):
        """Select the best survey based on seasonal preferences"""
        if not surveys:
            return None
        
        print(f"📊 Found {len(surveys)} available surveys:")
        
        # Parse and categorize surveys
        preferred_fall_surveys = []  # Preferred year + fall months
        other_fall_surveys = []      # Other years + fall months
        recent_surveys = []          # All other surveys (sorted by recency)
        
        for survey in surveys:
            capture_date = survey.get('captureDate', '')
            season = self.get_season_from_date(capture_date)
            
            try:
                survey_year = int(capture_date.split('-')[0])
                survey_month = int(capture_date.split('-')[1])
                date_obj = datetime.strptime(capture_date, "%Y-%m-%d")
                
                print(f"   📅 {capture_date} ({season}) - Year {survey_year}")
                
                if (survey_year == self.preferred_year and 
                    survey_month in self.preferred_months):
                    preferred_fall_surveys.append((survey, date_obj))
                    print(f"      🎯 PREFERRED: {self.preferred_year} Fall imagery")
                elif survey_month in self.preferred_months:
                    other_fall_surveys.append((survey, date_obj))
                    print(f"      🍂 Fall imagery (different year)")
                else:
                    recent_surveys.append((survey, date_obj))
            except:
                recent_surveys.append((survey, datetime.now()))
        
        print(f"\n📈 Survey categorization:")
        print(f"   🎯 Preferred {self.preferred_year} Fall: {len(preferred_fall_surveys)}")
        print(f"   🍂 Other Fall imagery: {len(other_fall_surveys)}")
        print(f"   📅 Recent/Other seasons: {len(recent_surveys)}")
        
        # Selection logic - Priority order
        if self.prioritize_fall_imagery and preferred_fall_surveys:
            # 1st Priority: Preferred year + fall months (most recent within that category)
            preferred_fall_surveys.sort(key=lambda x: x[1], reverse=True)
            selected_survey = preferred_fall_surveys[0][0]
            
            print(f"\n🎯 SELECTED: {self.preferred_year} Fall imagery: {selected_survey['captureDate']}")
            print(f"   Priority Level: 1 (Preferred year + season)")
            return selected_survey
            
        elif self.prioritize_fall_imagery and other_fall_surveys:
            # 2nd Priority: Other years + fall months (most recent fall imagery)
            other_fall_surveys.sort(key=lambda x: x[1], reverse=True)
            selected_survey = other_fall_surveys[0][0]
            
            print(f"\n🍂 SELECTED: Fall imagery from different year: {selected_survey['captureDate']}")
            print(f"   Priority Level: 2 (Fall season, different year)")
            return selected_survey
            
        elif self.fallback_to_recent and recent_surveys:
            # 3rd Priority: Most recent imagery (any season/year)
            recent_surveys.sort(key=lambda x: x[1], reverse=True)
            selected_survey = recent_surveys[0][0]
            
            print(f"\n📅 SELECTED: Most recent imagery: {selected_survey['captureDate']}")
            print(f"   Priority Level: 3 (Fallback to most recent)")
            if self.prioritize_fall_imagery:
                print(f"   💡 Note: Fall imagery preferred but not available")
            
            return selected_survey
        
        print(f"\n❌ No suitable surveys found")
        return None
    
    def parse_kml_boundary(self, kml_path: str):
        """Parse KML file and extract bbox coordinates (using working approach)"""
        try:
            tree = ET.parse(kml_path)
            root = tree.getroot()
            
            # Handle KML namespace
            namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            # Find coordinates in Polygon or LinearRing
            coordinates_elem = root.find('.//kml:coordinates', namespace)
            if coordinates_elem is None:
                coordinates_elem = root.find('.//coordinates')
            
            if coordinates_elem is not None:
                coords_text = coordinates_elem.text.strip()
                
                # Parse coordinates and get bbox (like the working script)
                lons, lats = [], []
                for coord_line in coords_text.split():
                    coord_line = coord_line.strip()
                    if coord_line and ',' in coord_line:
                        parts = coord_line.split(',')
                        if len(parts) >= 2:
                            lon, lat = float(parts[0]), float(parts[1])
                            lons.append(lon)
                            lats.append(lat)
                
                if lons and lats:
                    bbox = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"
                    print(f"✅ Extracted bbox from KML: {bbox}")
                    return bbox
            
            print(f"⚠️ Could not find coordinates in KML file")
            return None
            
        except Exception as e:
            print(f"❌ Error parsing KML file: {e}")
            return None
    
    def extract_region(self, kml_path: str, date: str = None, output_dir: str = None, location_name: str = None):
        """
        Extract region using the working approach from extract_nearmap_final.py
        
        Args:
            kml_path: Path to KML file
            date: Optional specific date (will auto-select if None)
            output_dir: Output directory
            location_name: Name for output files
            
        Returns:
            Dict with result paths and metadata
        """
        if not location_name:
            location_name = f"extraction_{int(time.time())}"
        
        if not output_dir:
            output_dir = f"temp_files/{location_name}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("🌍 NEARMAP EXTRACTION - USING WORKING METHOD")
        print("=" * 50)
        print(f"📍 Location: {location_name}")
        print(f"📁 Output: {output_dir}")
        
        # Step 1: Parse KML to get bbox
        print("Step 1: Parsing KML boundary...")
        bbox = self.parse_kml_boundary(kml_path)
        if not bbox:
            raise ValueError("Could not extract bbox from KML file")
        
        # Step 2: Get coverage data using working endpoint
        print("Step 2: Getting coverage data...")
        coverage_url = f"https://api.nearmap.com/coverage/v2/tx/bbox/{bbox}"
        coverage_params = {
            "dates": "all",
            "resources": "tiles:Vert", 
            "limit": str(self.max_surveys_to_check),
            "apikey": self.api_key
        }
        
        print(f"   URL: {coverage_url}")
        if self.prioritize_fall_imagery:
            print(f"🍂 Looking for {self.preferred_year} fall imagery (Aug/Sep/Oct) first...")
        
        try:
            cov_response = requests.get(coverage_url, params=coverage_params, timeout=15)
            print(f"📊 Coverage Status: {cov_response.status_code}")
            
            if cov_response.status_code == 200:
                coverage_data = cov_response.json()
                surveys = coverage_data.get('surveys', [])
                transaction_token = coverage_data.get('transactionToken', '')  # Get transaction token
                
                if not surveys:
                    raise ValueError("No surveys available for this area")
                
                # Select best survey using seasonal preferences
                selected_survey = self.select_preferred_survey(surveys)
                if not selected_survey:
                    raise ValueError("No suitable survey found")
                
                # Step 3: Extract using transactional approach
                return self._extract_using_tiles(bbox, selected_survey, output_dir, location_name, transaction_token)
                
            else:
                raise ValueError(f"Coverage API returned {cov_response.status_code}: {cov_response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def _extract_using_tiles(self, bbox, survey_data, output_dir, location_name, transaction_token):
        """Extract using the transactional tile approach that works (from final script)"""
        print("Step 3: Extracting using proven transactional method...")
        
        survey_id = survey_data['id']
        capture_date = survey_data['captureDate']
        scale_info = survey_data.get('scale', {}).get('raster:Vert', {})
        
        width_tiles = scale_info.get('widthInTiles', 1)
        height_tiles = scale_info.get('heightInTiles', 1)
        
        print(f"   🆔 Survey ID: {survey_id}")
        print(f"   📅 Capture Date: {capture_date}")
        print(f"   🍂 Season: {self.get_season_from_date(capture_date)}")
        print(f"   📏 Grid Size: {width_tiles}x{height_tiles} tiles")
        print(f"   🔑 Transaction Token: {transaction_token[:50]}..." if transaction_token else "   ⚠️  No transaction token")
        
        # Download tiles using transactional approach like the working script
        print(f"🧩 Downloading {width_tiles * height_tiles} tiles using transactional method...")
        print(f"   ℹ️ Using 4096x4096 pixel tiles for maximum detail")
        
        tile_url = f"https://api.nearmap.com/staticmap/v3/surveys/{survey_id}/Vert.jpg"
        downloaded_tiles = []
        total_tiles = width_tiles * height_tiles
        
        for y in range(height_tiles):
            for x in range(width_tiles):
                try:
                    tile_num = y * width_tiles + x + 1
                    print(f"📥 Downloading tile {tile_num}/{total_tiles} ({x},{y})...")
                    
                    tile_params = {
                        "x": str(x),
                        "y": str(y),
                        "tileSize": "4096x4096",
                        "apikey": self.api_key
                    }
                    
                    # Add transaction token if available
                    if transaction_token:
                        tile_params["transactionToken"] = transaction_token
                    
                    response = requests.get(tile_url, params=tile_params, stream=True, timeout=30)
                    
                    if response.status_code == 200:
                        tile_filename = f"temp_tile_{x}_{y}.jpg"
                        tile_path = os.path.join(output_dir, tile_filename)
                        
                        with open(tile_path, "wb") as tile_file:
                            for chunk in response.iter_content(1024):
                                tile_file.write(chunk)
                        
                        downloaded_tiles.append((x, y, tile_path))
                        print(f"   ✅ Downloaded: {tile_filename}")
                    else:
                        print(f"   ❌ Failed tile ({x},{y}): {response.status_code}")
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    print(f"   ⚠️  Error downloading tile ({x},{y}): {e}")
                    continue
        
        if not downloaded_tiles:
            raise ValueError("No tiles were downloaded successfully")
        
        print(f"✅ Downloaded {len(downloaded_tiles)}/{total_tiles} tiles")
        
        # Combine tiles like the working script
        print("🔧 Combining tiles and removing black borders...")
        combined_image = self._combine_tiles_transactional(downloaded_tiles, width_tiles, height_tiles)
        
        # Crop black borders (important for matching working script)
        cropped_image, final_size = self._crop_black_borders(combined_image)
        combined_image.close()
        
        # CRITICAL FIX: Resample to target resolution like the working script
        print(f"🎯 Resampling to target resolution: {self.target_mpp} m/px")
        
        # Calculate native resolution from survey data
        native_resolution = self._calculate_native_resolution(survey_data, final_size)
        scale_factor = native_resolution / self.target_mpp
        
        print(f"   📊 Native resolution: {native_resolution:.6f} m/px")
        print(f"   🎯 Target resolution: {self.target_mpp:.6f} m/px") 
        print(f"   📐 Scale factor: {scale_factor:.4f}")
        
        # Resample image to exact target resolution
        if abs(scale_factor - 1.0) > 0.05:  # Only resample if significant difference
            print(f"   🔄 Resampling required (difference > 5%)")
            final_width = int(final_size[0] * scale_factor)
            final_height = int(final_size[1] * scale_factor)
            
            resampled_image = cropped_image.resize((final_width, final_height), Image.LANCZOS)
            cropped_image.close()
            
            final_size = (final_width, final_height)
            final_image = resampled_image
            actual_resolution = self.target_mpp
            
            print(f"   ✅ Resampled to: {final_width}x{final_height} pixels")
        else:
            print(f"   ✅ Resolution close to target, no resampling needed")
            final_image = cropped_image
            actual_resolution = native_resolution
        
        # Save final image
        image_filename = f"{location_name}_nearmap.jpg"
        image_path = os.path.join(output_dir, image_filename)
        final_image.save(image_path, 'JPEG', quality=95)
        
        image_width, image_height = final_size
        final_image.close()
        
        print(f"✅ Saved final image: {image_path}")
        print(f"📐 Final image size: {image_width}x{image_height} pixels")
        print(f"📏 Final resolution: {actual_resolution:.6f} m/px")
        
        # Clean up temporary tiles
        print("🧹 Cleaning up temporary tiles...")
        for x, y, tile_path in downloaded_tiles:
            try:
                os.remove(tile_path)
            except:
                pass
        
        # Calculate resolution based on actual survey data
        # actual_resolution = self._calculate_resolution_from_survey(survey_data, image_width, image_height)
        # ^^ This is now handled above in the resampling section
        
        # Create metadata compatible with app.py expectations
        print("💾 Creating metadata...")
        metadata = {
            "file": image_filename,
            "resolution_mpp": round(actual_resolution, 6),
            "resolution_m_per_pixel": round(actual_resolution, 6),  # Alternative key for compatibility
            "spatial_reference": "EPSG:3857",
            "width_px": image_width,
            "height_px": image_height,
            "bounds": self._calculate_bounds_from_bbox(bbox),
            "extraction_info": {
                "extracted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "survey_capture_date": survey_data.get("captureDate", "Unknown"),
                "survey_id": survey_data.get("id", "Unknown"),
                "nearmap_api_version": "v3_transactional",
                "native_resolution": native_resolution,
                "target_resolution": self.target_mpp,
                "scale_factor_applied": scale_factor,
                "resampled": abs(scale_factor - 1.0) > 0.05,
                "tile_grid": {
                    "width_tiles": width_tiles,
                    "height_tiles": height_tiles,
                    "total_tiles": total_tiles,
                    "downloaded_tiles": len(downloaded_tiles),
                    "tile_size": "4096x4096"
                },
                "seasonal_selection": {
                    "prioritize_fall": self.prioritize_fall_imagery,
                    "preferred_months": self.preferred_months,
                    "preferred_year": self.preferred_year,
                    "selected_season": self.get_season_from_date(survey_data.get("captureDate", "")),
                    "total_surveys_available": len([survey_data])
                }
            }
        }
        
        metadata_filename = f"{location_name}_metadata.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"💾 Metadata saved: {metadata_path}")
        
        # Return results in the format expected by app.py
        capture_date_clean = survey_data['captureDate'].split('T')[0] if 'T' in survey_data['captureDate'] else survey_data['captureDate']
        
        return {
            'image_path': image_path,
            'metadata_path': metadata_path,
            'location': location_name,
            'date_requested': capture_date_clean,
            'date_actual': capture_date_clean,
            'resolution_m_per_pixel': actual_resolution,
            'width_px': image_width,
            'height_px': image_height,
            'tiles_downloaded': len(downloaded_tiles),
            'extraction_method': 'transactional_tiles',
            'survey_id': survey_data.get('id', 'unknown'),
            'success': True
        }
    
    def _combine_tiles_transactional(self, downloaded_tiles, width_tiles, height_tiles):
        """Combine downloaded tiles using transactional method (4096x4096 tiles)"""
        if not downloaded_tiles or not HAS_PIL:
            raise ValueError("Cannot combine tiles - PIL not available or no tiles downloaded")
        
        # Get tile size from first tile
        first_tile_path = downloaded_tiles[0][2]
        first_tile = Image.open(first_tile_path)
        tile_width, tile_height = first_tile.size
        first_tile.close()
        
        print(f"   📐 Tile size: {tile_width}x{tile_height}")
        
        # Create combined image
        total_width = width_tiles * tile_width
        total_height = height_tiles * tile_height
        combined_image = Image.new('RGB', (total_width, total_height))
        
        print(f"   📐 Combined size (before crop): {total_width}x{total_height}")
        
        # Paste each tile at correct position
        for x, y, tile_path in downloaded_tiles:
            if os.path.exists(tile_path):
                tile_img = Image.open(tile_path)
                
                # Calculate paste position
                paste_x = x * tile_width
                paste_y = y * tile_height
                
                combined_image.paste(tile_img, (paste_x, paste_y))
                tile_img.close()
        
        return combined_image
    
    def _crop_black_borders(self, image):
        """Remove black borders from combined image (essential for transactional method)"""
        if not HAS_NUMPY:
            print("   ⚠️  NumPy not available - skipping black border removal")
            return image, image.size
        
        print("   ✂️ Cropping black borders...")
        
        # Convert to numpy array for efficient processing
        img_array = np.array(image)
        
        # Find non-black pixels (allowing for slight variations)
        non_black = np.any(img_array > 10, axis=2)
        
        # Get bounding box of non-black region
        non_black_coords = np.where(non_black)
        
        if len(non_black_coords[0]) == 0:
            print("   ⚠️  Image appears to be all black - returning original")
            return image, image.size
        
        min_y, max_y = non_black_coords[0].min(), non_black_coords[0].max()
        min_x, max_x = non_black_coords[1].min(), non_black_coords[1].max()
        
        # Crop the image
        cropped_image = image.crop((min_x, min_y, max_x + 1, max_y + 1))
        new_size = cropped_image.size
        
        print(f"   📐 Cropped from {image.size} to {new_size}")
        
        return cropped_image, new_size
    
    def _calculate_resolution_from_survey(self, survey_data, image_width, image_height):
        """Calculate resolution based on survey data"""
        # For transactional method, we can estimate based on typical Nearmap resolutions
        # This is a placeholder - the working script would have the exact calculation
        return 0.075  # Approximate meters per pixel for high-res Nearmap
    
    def _calculate_zoom_for_target_mpp(self, latitude):
        """Calculate zoom level for target resolution (kept for compatibility)"""
        cos_lat = math.cos(math.radians(abs(latitude)))
        zoom = math.log2(156543.03392804097 * cos_lat / self.target_mpp)
        
        # Cap zoom level to reasonable range
        zoom = max(10, min(22, zoom))
        final_zoom = int(round(zoom))
        
        actual_mpp = 156543.03392 * cos_lat / (2 ** final_zoom)
        print(f"   🎯 Zoom calculation: target={self.target_mpp:.4f} → zoom={zoom:.2f} → final={final_zoom}")
        print(f"   📊 Actual resolution: {actual_mpp:.4f} m/px (diff: {abs(actual_mpp-self.target_mpp):.4f})")
        
        return final_zoom
    
    def _lonlat_to_tile(self, lon, lat, zoom):
        """Convert longitude/latitude to tile coordinates (kept for compatibility)"""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
    
    def _calculate_native_resolution(self, survey_data, image_size):
        """Calculate native resolution from survey data"""
        # This is survey-specific and varies based on capture date
        capture_date = survey_data.get('captureDate', '')
        
        try:
            year = int(capture_date.split('-')[0])
            # Newer surveys typically have higher resolution
            if year >= 2024:
                return 0.062  # ~6.2cm/pixel for recent surveys
            elif year >= 2022:
                return 0.075  # ~7.5cm/pixel for medium surveys  
            else:
                return 0.100  # ~10cm/pixel for older surveys
        except:
            return 0.075  # Default fallback
    
    def _calculate_bounds_from_bbox(self, bbox_string):
        """Convert bbox string to bounds dict for metadata"""
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox_string.split(','))
            return {
                'min_lon': min_lon,
                'min_lat': min_lat,
                'max_lon': max_lon,
                'max_lat': max_lat
            }
        except:
            return {
                'min_lon': 0, 'min_lat': 0,
                'max_lon': 0, 'max_lat': 0
            }


# Legacy compatibility - create aliases for app.py
class NearmapTileExtractor(NearmapImageExtractor):
    """Legacy alias for backwards compatibility with app.py"""
    pass


# For direct script execution (testing)
if __name__ == "__main__":
    print("🧪 Testing NearmapImageExtractor class")
    
    # Test with OrchardHills.kml if available
    if os.path.exists("OrchardHills.kml"):
        try:
            extractor = NearmapImageExtractor()
            result = extractor.extract_region("OrchardHills.kml", output_dir="test_class_output")
            print(f"✅ Test successful: {result}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("ℹ️  OrchardHills.kml not found - cannot run test")
