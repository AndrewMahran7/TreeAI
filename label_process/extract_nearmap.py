import os
import requests
import json
import math
import time
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import pyproj
from typing import List, Tuple, Dict, Any, Set
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import functools

# Override PIL's MAX_IMAGE_PIXELS limit to handle very large images
Image.MAX_IMAGE_PIXELS = None

class NearmapTileExtractor:
    """
    Extract and combine Nearmap tiles based on KML boundary and date.
    - Ensures ~3 inch (~0.075 m/px) spatial resolution by selecting zoom dynamically.
    - Downloads and combines ONLY tiles that intersect the KML geofence (no extras).
    """
    
    def __init__(self, api_key: str = None, target_mpp: float = 0.075):
        """
        Initialize the Nearmap tile extractor.
        
        Args:
            api_key (str): Nearmap API key. If None, will look for NEARMAP_API_KEY env var.
            target_mpp (float): Desired meters-per-pixel (defaults to ~3 inch = 0.075 m/px).
        """
        self.api_key = api_key or os.getenv('NEARMAP_API_KEY')
        if not self.api_key:
            raise ValueError("Nearmap API key required. Set NEARMAP_API_KEY env var or pass as parameter.")
        
        # Nearmap tile server settings
        self.base_url = "https://api.nearmap.com/tiles/v3/Vert"
        self.target_mpp = target_mpp
        
        # Determine optimal tile size based on target resolution
        # Nearmap supports multiple tile sizes: 256, 512, 1024, 2048, etc.
        # Larger tiles mean fewer requests but more memory usage
        self.tile_size = self.get_optimal_tile_size(target_mpp)
        
        # Coordinate system transformations
        self.wgs84_to_mercator = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.mercator_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        
        # Store polygon for intersection testing
        self.boundary_polygon: Polygon | None = None

    # ---------------------------
    # KML / Geometry
    # ---------------------------
    def parse_kml_boundary(self, kml_path: str) -> List[Tuple[float, float]]:
        """
        Parse KML file to extract boundary coordinates (first polygon encountered).
        
        Args:
            kml_path (str): Path to KML file
            
        Returns:
            List[Tuple[float, float]]: List of (lon, lat) coordinate pairs
        """
        try:
            tree = ET.parse(kml_path)
            root = tree.getroot()
            
            # Handle KML namespace
            namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            # Try namespaced first, then fallback
            coords_element = root.find('.//kml:coordinates', namespace)
            if coords_element is None:
                coords_element = root.find('.//coordinates')
            if coords_element is None or not coords_element.text:
                raise ValueError("No coordinates found in KML file")
            
            coords_text = coords_element.text.strip()
            coordinates: List[Tuple[float, float]] = []
            for coord_str in coords_text.split():
                parts = coord_str.split(',')
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    coordinates.append((lon, lat))
            
            if len(coordinates) < 3:
                raise ValueError("KML polygon must contain at least 3 coordinates.")
            
            # Ensure polygon closure
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            
            print(f"✅ Parsed {len(coordinates)} coordinates from KML")
            return coordinates
            
        except Exception as e:
            print(f"❌ Error parsing KML file: {e}")
            raise
    
    def get_bounding_box(self, coordinates: List[Tuple[float, float]]) -> Dict[str, float]:
        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]
        return {'min_lon': min(lons), 'max_lon': max(lons), 'min_lat': min(lats), 'max_lat': max(lats)}
    
    def create_polygon_from_coordinates(self, coordinates: List[Tuple[float, float]]) -> Polygon:
        return Polygon(coordinates)
    
    # ---------------------------
    # Web Mercator tiling helpers
    # ---------------------------
    def lonlat_to_tile(self, lon: float, lat: float, zoom: int) -> Tuple[int, int]:
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    def tile_to_lonlat(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return (lon, lat)

    def get_tile_bounds(self, x: int, y: int, zoom: int) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        tl_lon, tl_lat = self.tile_to_lonlat(x, y, zoom)
        tr_lon, tr_lat = self.tile_to_lonlat(x + 1, y, zoom)
        br_lon, br_lat = self.tile_to_lonlat(x + 1, y + 1, zoom)
        bl_lon, bl_lat = self.tile_to_lonlat(x, y + 1, zoom)
        return ((tl_lon, tl_lat), (tr_lon, tr_lat), (br_lon, br_lat), (bl_lon, bl_lat))

    def calculate_tile_range(self, bbox: Dict[str, float], zoom: int) -> Dict[str, int]:
        min_tile_x, max_tile_y = self.lonlat_to_tile(bbox['min_lon'], bbox['min_lat'], zoom)
        max_tile_x, min_tile_y = self.lonlat_to_tile(bbox['max_lon'], bbox['max_lat'], zoom)
        return {'min_x': min_tile_x, 'max_x': max_tile_x, 'min_y': min_tile_y, 'max_y': max_tile_y}

    # ---------------------------
    # Polygon intersection
    # ---------------------------
    def tile_intersects_polygon(self, x: int, y: int, zoom: int, polygon: Polygon) -> bool:
        corners = self.get_tile_bounds(x, y, zoom)
        tile_polygon = Polygon(corners)
        return polygon.intersects(tile_polygon)
    
    def get_intersecting_tiles(self, bbox: Dict[str, float], zoom: int, polygon: Polygon) -> Set[Tuple[int, int]]:
        tile_range = self.calculate_tile_range(bbox, zoom)
        intersecting_tiles: Set[Tuple[int, int]] = set()
        
        print(f"   Checking tiles for polygon intersection...")
        total_tiles = (tile_range['max_x'] - tile_range['min_x'] + 1) * (tile_range['max_y'] - tile_range['min_y'] + 1)
        checked = 0
        
        for tile_x in range(tile_range['min_x'], tile_range['max_x'] + 1):
            for tile_y in range(tile_range['min_y'], tile_range['max_y'] + 1):
                checked += 1
                if checked % 50 == 0:
                    print(f"   Checked {checked}/{total_tiles} tiles...")
                if self.tile_intersects_polygon(tile_x, tile_y, zoom, polygon):
                    intersecting_tiles.add((tile_x, tile_y))
        
        print(f"   Found {len(intersecting_tiles)}/{total_tiles} tiles intersecting polygon")
        return intersecting_tiles

    # ---------------------------
    # Resolution helpers
    # ---------------------------
    @staticmethod
    def meters_per_pixel_at_lat(zoom: int, lat_deg: float) -> float:
        # Standard Web Mercator ground resolution formula
        return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)
    
    @staticmethod
    def calculate_actual_resolution(min_lon: float, max_lon: float, min_lat: float, max_lat: float, width_px: int, height_px: int) -> float:
        """
        Calculate actual spatial resolution based on geographic bounds and image dimensions.
        This gives the true meters-per-pixel based on actual geographic distance.
        """
        # Calculate geographic distance in meters using great circle distance
        # Using Haversine formula for accuracy
        lat1, lat2 = math.radians(min_lat), math.radians(max_lat)
        lon1, lon2 = math.radians(min_lon), math.radians(max_lon)
        
        # Calculate horizontal distance (longitude difference at average latitude)
        avg_lat = (lat1 + lat2) / 2
        dlon = lon2 - lon1
        horizontal_distance = 6371000 * dlon * math.cos(avg_lat)  # Earth radius = 6371000m
        
        # Calculate vertical distance (latitude difference)
        dlat = lat2 - lat1
        vertical_distance = 6371000 * dlat  # Earth radius = 6371000m
        
        # Calculate resolution in both directions
        horizontal_resolution = abs(horizontal_distance) / width_px
        vertical_resolution = abs(vertical_distance) / height_px
        
        # Return the average resolution
        return (horizontal_resolution + vertical_resolution) / 2

    @classmethod
    def zoom_for_target_mpp(cls, target_mpp: float, lat_deg: float) -> int:
        """
        Choose the zoom level whose meters-per-pixel is closest to the target_mpp.
        For 3-inch resolution (~0.075 m/px), this will select the optimal zoom level.
        """
        best_zoom = 0
        best_diff = float("inf")
        for z in range(0, 24):  # Nearmap supports 0–23
            mpp = cls.meters_per_pixel_at_lat(z, lat_deg)
            diff = abs(mpp - target_mpp)
            if diff < best_diff:
                best_zoom, best_diff = z, diff
        
        # For any location, find zoom level closest to target without going too high
        # Most locations at mid-latitudes (25-50°) should use optimal zoom for target resolution
        if 25 <= lat_deg <= 50 and target_mpp <= 0.15:  # Mid-latitude range (most populated areas)
            # Find the zoom level that gives us closest to target, preferring slightly under target
            best_candidate_zoom = best_zoom
            best_candidate_diff = best_diff
            
            for candidate_zoom in [18, 19, 20, 21]:  # Check zoom levels in order of preference
                candidate_mpp = cls.meters_per_pixel_at_lat(candidate_zoom, lat_deg)
                candidate_diff = abs(candidate_mpp - target_mpp)
                
                # Prefer zoom levels that are close to target and not too much higher resolution
                if candidate_mpp <= target_mpp * 1.3 and candidate_diff < best_candidate_diff:
                    best_candidate_zoom = candidate_zoom
                    best_candidate_diff = candidate_diff
            
            print(f"   🎯 Using zoom {best_candidate_zoom} for target resolution: {cls.meters_per_pixel_at_lat(best_candidate_zoom, lat_deg):.4f} m/px (target: {target_mpp:.4f} m/px)")
            return best_candidate_zoom
        
        return best_zoom


    # ---------------------------
    # Configuration helpers
    # ---------------------------
    def get_optimal_tile_size(self, target_mpp: float) -> int:
        """
        Get the largest practical tile size for the target resolution.
        Larger tiles mean fewer API requests and better performance.
        """
        # Available tile sizes in Nearmap (powers of 2)
        available_sizes = [256, 512, 1024, 2048]
        
        # For higher resolution imagery (smaller m/px), use larger tiles
        # This reduces the number of API calls needed
        if target_mpp <= 0.05:  # Very high resolution
            tile_size = 2048
        elif target_mpp <= 0.1:   # High resolution  
            tile_size = 1024
        elif target_mpp <= 0.2:   # Medium resolution
            tile_size = 512
        else:                     # Lower resolution
            tile_size = 256
            
        print(f"🎯 Selected tile size: {tile_size}x{tile_size} pixels for {target_mpp:.3f} m/px resolution")
        return tile_size
    
    def get_latest_imagery_date_and_transaction(self, bbox_coords: List[Tuple[float, float]]) -> Tuple[str, str, Dict]:
        """
        Get the latest available imagery date and transaction token using the transactional method.
        Returns date, transaction_token, and survey data for efficient bulk downloading.
        """
        # Convert coordinates to bbox format
        lons = [coord[0] for coord in bbox_coords]
        lats = [coord[1] for coord in bbox_coords]
        bbox_str = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"
        
        coverage_url = f"https://api.nearmap.com/coverage/v2/tx/bbox/{bbox_str}"
        coverage_params = {
            "dates": "all",
            "resources": "tiles:Vert", 
            "limit": "10",  # Get more surveys to choose from
            "apikey": self.api_key
        }
        
        try:
            print("🔍 Getting transactional coverage data...")
            print(f"   📦 Bbox: {bbox_str}")
            
            response = requests.get(coverage_url, params=coverage_params, timeout=30)
            
            if response.status_code == 200:
                coverage_data = response.json()
                surveys = coverage_data.get('surveys', [])
                transaction_token = coverage_data.get('transactionToken', '')
                
                if surveys and transaction_token:
                    print(f"   ✅ Found {len(surveys)} available surveys with transaction token")
                    
                    # Select best survey using the same seasonal logic
                    best_survey = self._select_best_survey_from_list(surveys)
                    best_date = best_survey['captureDate'].split('T')[0]
                    
                    print(f"   🎯 Selected survey: {best_survey['id']} from {best_date}")
                    print(f"   🔑 Transaction token: {transaction_token[:50]}...")
                    
                    return best_date, transaction_token, best_survey
                else:
                    print("⚠️ No surveys or transaction token in transactional response")
                    
            else:
                print(f"⚠️ Transactional coverage API returned {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Transactional coverage API request error: {e}")
        except Exception as e:
            print(f"⚠️ Error getting transactional coverage: {e}")
        
        # Fallback to regular coverage method
        print("🔄 Falling back to regular coverage method...")
        center_lat = sum(coord[1] for coord in bbox_coords) / len(bbox_coords)
        center_lon = sum(coord[0] for coord in bbox_coords) / len(bbox_coords)
        fallback_date = self.get_latest_imagery_date_fallback(center_lat, center_lon)
        return fallback_date, None, None
    
    def get_latest_imagery_date_fallback(self, lat: float, lon: float) -> str:
        """
        Fallback method for getting imagery date when transactional method fails.
        """
        coverage_url = "https://api.nearmap.com/coverage/v2/poly"
        
        # Create a small polygon around the point for coverage check
        buffer = 0.001  # Small buffer around the point
        polygon_coords = [
            [lon - buffer, lat - buffer],
            [lon + buffer, lat - buffer], 
            [lon + buffer, lat + buffer],
            [lon - buffer, lat + buffer],
            [lon - buffer, lat - buffer]
        ]
        
        payload = {
            'polygon': json.dumps(polygon_coords),
            'apikey': self.api_key
        }
        
        try:
            print("🔍 Checking for available imagery...")
            response = requests.get(coverage_url, params=payload, timeout=30)
            
            if response.status_code == 200:
                coverage_data = response.json()
                
                if 'surveys' in coverage_data and coverage_data['surveys']:
                    surveys = coverage_data['surveys']
                    print(f"   ✅ Found {len(surveys)} available surveys")
                    return self._select_best_seasonal_date(surveys)
                else:
                    print("⚠️ No coverage data found for this location")
            elif response.status_code == 404:
                print("⚠️ Coverage endpoint not found - trying alternative endpoints...")
                return self._try_alternative_coverage_endpoints(lat, lon)
            else:
                print(f"⚠️ Coverage API returned {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Coverage API request error: {e}")
        except Exception as e:
            print(f"⚠️ Error checking imagery coverage: {e}")
        
        # All coverage methods failed, use intelligent fallback
        print("🔄 All coverage methods failed - using intelligent seasonal fallback")
        return self._get_intelligent_fallback_date()
    
    def _try_alternative_coverage_endpoints(self, lat: float, lon: float) -> str:
        """Try alternative coverage API endpoints if the main one fails"""
        alternative_endpoints = [
            "https://api.nearmap.com/coverage/v2/polygon",
            "https://api.nearmap.com/coverage/v1/poly", 
            "https://api.nearmap.com/coverage"
        ]
        
        buffer = 0.001
        polygon_coords = [
            [lon - buffer, lat - buffer],
            [lon + buffer, lat - buffer], 
            [lon + buffer, lat + buffer],
            [lon - buffer, lat + buffer],
            [lon - buffer, lat - buffer]
        ]
        
        payload = {
            'polygon': json.dumps(polygon_coords),
            'apikey': self.api_key
        }
        
        for i, endpoint in enumerate(alternative_endpoints, 1):
            try:
                print(f"   Trying endpoint {i}/{len(alternative_endpoints)}: {endpoint}")
                response = requests.get(endpoint, params=payload, timeout=15)
                
                if response.status_code == 200:
                    coverage_data = response.json()
                    if 'surveys' in coverage_data and coverage_data['surveys']:
                        surveys = coverage_data['surveys']
                        print(f"   ✅ Success with {endpoint}! Found {len(surveys)} surveys")
                        return self._select_best_seasonal_date(surveys)
                else:
                    print(f"   ❌ {endpoint} returned {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error with {endpoint}: {e}")
                continue
        
        # All alternative endpoints failed
        return self._get_intelligent_fallback_date()
    
    def _select_best_survey_from_list(self, surveys):
        """Select the best survey from a list using seasonal preferences"""
        print(f"📊 Processing {len(surveys)} available surveys...")
        
        # Generate comprehensive preferred dates (Aug 1 - Oct 31 for past 3 years)
        current_year = datetime.now().year
        preferred_dates = []
        
        # Priority order: Past 3 years, seasonal months (Aug-Oct), each day
        for year in [current_year-1, current_year-2, current_year-3]:
            for month in [9, 8, 10]:  # September first (best for trees), then August, October
                # Check every day in the seasonal months
                days_in_month = {8: 31, 9: 30, 10: 31}[month]
                for day in range(1, days_in_month + 1):
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    preferred_dates.append(date_str)
        
        print(f"   🔍 Checking {len(preferred_dates)} preferred seasonal dates across 3 years...")
        
        # Find the best available survey matching our date preferences
        best_survey = None
        best_priority = float('inf')
        all_valid_surveys = []
        
        for survey in surveys:
            survey_date = survey.get('captureDate', '')
            if survey_date:
                full_survey_date = survey_date.split('T')[0]  # Get YYYY-MM-DD
                
                try:
                    survey_datetime = datetime.strptime(full_survey_date, '%Y-%m-%d')
                    
                    # Add to valid surveys list (for fallback)
                    all_valid_surveys.append({
                        'survey': survey, 
                        'date': full_survey_date, 
                        'datetime': survey_datetime
                    })
                    
                    # Check if this date is in our preferred list
                    if full_survey_date in preferred_dates:
                        # Safety check: Only use dates that are definitely in the past
                        if survey_datetime <= datetime.now() - timedelta(days=30):
                            priority = preferred_dates.index(full_survey_date)
                            if priority < best_priority:
                                best_priority = priority
                                best_survey = survey
                                print(f"   ✅ Found preferred date: {full_survey_date} (priority: {priority})")
                            
                except ValueError as e:
                    print(f"   ⚠️ Invalid date format in survey: {survey_date}")
                    continue
        
        # If we found a preferred seasonal date, use it
        if best_survey:
            return best_survey
        
        # No preferred seasonal date found - look for any seasonal dates
        seasonal_surveys = []
        for valid_survey in all_valid_surveys:
            survey_date = valid_survey['date']
            survey_month = int(survey_date.split('-')[1])
            
            # Check if it's in seasonal months (Aug-Oct) from any year
            if survey_month in [8, 9, 10]:
                seasonal_surveys.append(valid_survey)
        
        if seasonal_surveys:
            # Sort seasonal surveys by date (most recent first)
            seasonal_surveys.sort(key=lambda x: x['datetime'], reverse=True)
            return seasonal_surveys[0]['survey']
        
        # No seasonal imagery found - use most recent available
        if all_valid_surveys:
            all_valid_surveys.sort(key=lambda x: x['datetime'], reverse=True)
            return all_valid_surveys[0]['survey']
            
        # Should not reach here, but return first survey if available
        return surveys[0] if surveys else None
        """Select the best seasonal date from available surveys with comprehensive date checking"""
        print(f"📊 Processing {len(surveys)} available surveys...")
        
        # Generate comprehensive preferred dates (Aug 1 - Oct 31 for past 3 years)
        current_year = datetime.now().year
        preferred_dates = []
        
        # Priority order: Past 3 years, seasonal months (Aug-Oct), each day
        for year in [current_year-1, current_year-2, current_year-3]:
            for month in [9, 8, 10]:  # September first (best for trees), then August, October
                # Check every day in the seasonal months
                days_in_month = {8: 31, 9: 30, 10: 31}[month]
                for day in range(1, days_in_month + 1):
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    preferred_dates.append(date_str)
        
        print(f"   🔍 Checking {len(preferred_dates)} preferred seasonal dates across 3 years...")
        
        # Find the best available survey matching our comprehensive date preferences
        best_survey = None
        best_priority = float('inf')
        all_valid_surveys = []  # Track all valid surveys for fallback
        
        for survey in surveys:
            survey_date = survey.get('captureDate', '')
            if survey_date:
                full_survey_date = survey_date.split('T')[0]  # Get YYYY-MM-DD
                
                # Parse survey date to check if it's valid and in the past
                try:
                    survey_datetime = datetime.strptime(full_survey_date, '%Y-%m-%d')
                    
                    # Add to valid surveys list (we'll use this for fallback)
                    all_valid_surveys.append({
                        'survey': survey, 
                        'date': full_survey_date, 
                        'datetime': survey_datetime
                    })
                    
                    # Check if this date is in our preferred list
                    if full_survey_date in preferred_dates:
                        # Safety check: Only use dates that are definitely in the past
                        if survey_datetime <= datetime.now() - timedelta(days=30):
                            priority = preferred_dates.index(full_survey_date)
                            if priority < best_priority:
                                best_priority = priority
                                best_survey = survey
                                print(f"   ✅ Found preferred date: {full_survey_date} (priority: {priority})")
                            
                except ValueError as e:
                    print(f"   ⚠️ Invalid date format in survey: {survey_date}")
                    continue
        
        # If we found a preferred seasonal date, use it
        if best_survey:
            best_date = best_survey['captureDate'].split('T')[0]
            survey_month = int(best_date.split('-')[1])
            season_name = {8: "August", 9: "September", 10: "October"}.get(survey_month, f"Month {survey_month}")
            print(f"✅ Selected optimal seasonal imagery: {best_date} ({season_name} - perfect for tree detection)")
            return best_date
        
        # No preferred seasonal date found - look for any seasonal dates (current year or other years)
        seasonal_surveys = []
        for valid_survey in all_valid_surveys:
            survey_date = valid_survey['date']
            survey_month = int(survey_date.split('-')[1])
            
            # Check if it's in seasonal months (Aug-Oct) from any year
            if survey_month in [8, 9, 10]:
                seasonal_surveys.append(valid_survey)
        
        if seasonal_surveys:
            # Sort seasonal surveys by date (most recent first)
            seasonal_surveys.sort(key=lambda x: x['datetime'], reverse=True)
            most_recent_seasonal = seasonal_surveys[0]
            seasonal_date = most_recent_seasonal['date']
            
            survey_month = int(seasonal_date.split('-')[1])
            season_name = {8: "August", 9: "September", 10: "October"}[survey_month]
            
            print(f"✅ Using seasonal imagery: {seasonal_date} ({season_name} - good for tree detection)")
            return seasonal_date
        
        # No seasonal imagery found at all - use most recent available
        if all_valid_surveys:
            # Sort by date (most recent first)
            all_valid_surveys.sort(key=lambda x: x['datetime'], reverse=True)
            most_recent = all_valid_surveys[0]
            most_recent_date = most_recent['date']
            
            survey_month = int(most_recent_date.split('-')[1])
            month_name = datetime.strptime(most_recent_date, '%Y-%m-%d').strftime('%B')
            
            if survey_month in [8, 9, 10]:
                print(f"✅ Using most recent seasonal imagery: {most_recent_date} ({month_name})")
            else:
                print(f"⚠️  Using most recent available imagery: {most_recent_date} ({month_name})")
                print(f"   No seasonal imagery (Aug-Oct) found - tree detection may be less accurate")
                
            return most_recent_date
            
        # Should not reach here, but fallback anyway
        print("❌ No valid surveys found in coverage data")
        return self._get_intelligent_fallback_date()
    
    def _get_intelligent_fallback_date(self):
        """Get intelligently selected date when coverage API is completely unavailable"""
        current_year = datetime.now().year
        
        print("🎯 Using intelligent comprehensive date selection...")
        
        # Build comprehensive priority list: Aug 1 - Oct 31 for past 3 years
        priority_dates = []
        
        # Priority order: Past 3 years, each day in seasonal months
        for year in [current_year-1, current_year-2, current_year-3]:
            for month in [9, 8, 10]:  # September first (best), August, October
                days_in_month = {8: 31, 9: 30, 10: 31}[month]
                for day in range(1, days_in_month + 1):
                    try:
                        test_date = datetime(year, month, day)
                        # Only use dates that are at least 60 days old to ensure imagery is available
                        if test_date <= datetime.now() - timedelta(days=60):
                            priority_dates.append(test_date.strftime('%Y-%m-%d'))
                    except ValueError:  # Handle any edge cases
                        continue
        
        # Second priority: Current year seasonal dates (if old enough)
        for month in [9, 8, 10]:  # September, August, October
            days_in_month = {8: 31, 9: 30, 10: 31}[month]
            for day in range(1, days_in_month + 1):
                try:
                    test_date = datetime(current_year, month, day)
                    # Only use current year dates if they're at least 30 days old
                    if test_date <= datetime.now() - timedelta(days=30):
                        priority_dates.append(test_date.strftime('%Y-%m-%d'))
                except ValueError:  # Handle invalid dates
                    continue
        
        # Third priority: Other months from past years (non-seasonal but still useful)
        for year in [current_year-1, current_year-2]:
            for month in [7, 11, 6, 5]:  # July, November, June, May
                for day in [15]:  # Just mid-month to avoid too many dates
                    try:
                        test_date = datetime(year, month, day)
                        if test_date <= datetime.now() - timedelta(days=60):
                            priority_dates.append(test_date.strftime('%Y-%m-%d'))
                    except ValueError:
                        continue
        
        if priority_dates:
            selected_date = priority_dates[0]  # Use the first (highest priority) date
            selected_year = int(selected_date[:4])
            selected_month = int(selected_date[5:7])
            selected_day = int(selected_date[8:10])
            
            month_name = datetime(selected_year, selected_month, selected_day).strftime('%B')
            
            # Categorize the selected date
            if selected_month in [8, 9, 10]:
                season_quality = "🍂 Optimal seasonal"
            elif selected_month in [7, 11]:
                season_quality = "🌿 Good seasonal"
            else:
                season_quality = "📅 Off-season"
            
            print(f"📅 Intelligent fallback: {selected_date} ({season_quality} - {month_name} {selected_year})")
            
            if selected_year == current_year:
                print(f"   ⚠️ Using current year date - imagery might be very recent")
            else:
                print(f"   ✅ Using historical date - imagery should be well-established")
                
            return selected_date
        else:
            # Absolute last resort - use a known working date
            fallback_date = "2024-08-26"  # August 26, 2024 - your confirmed working date
            print(f"🛟 Using emergency fallback: {fallback_date} (known working date)")
            return fallback_date

    # ---------------------------
    # Transactional downloading (preferred method)
    # ---------------------------
    def download_tiles_transactional(self, survey_id: str, transaction_token: str, width_tiles: int, height_tiles: int, output_dir: str) -> List[Tuple[int, int, str]]:
        """
        Download tiles using the transactional method - more efficient and reliable
        """
        tile_url = f"https://api.nearmap.com/staticmap/v3/surveys/{survey_id}/Vert.jpg"
        tiles = []
        total_tiles = width_tiles * height_tiles
        
        print(f"🧩 Downloading {total_tiles} tiles using transactional method...")
        
        for y in range(height_tiles):
            for x in range(width_tiles):
                tile_num = y * width_tiles + x + 1
                print(f"📥 Downloading tile {tile_num}/{total_tiles} ({x},{y})...")
                
                tile_params = {
                    "x": str(x),
                    "y": str(y),
                    "tileSize": f"{self.tile_size}x{self.tile_size}",
                    "transactionToken": transaction_token,
                    "apikey": self.api_key
                }
                
                tile_filename = f"tile_tx_{x}_{y}.jpg"
                tile_path = os.path.join(output_dir, tile_filename)
                
                try:
                    tile_response = requests.get(tile_url, params=tile_params, stream=True, timeout=30)
                    
                    if tile_response.status_code == 200:
                        with open(tile_path, "wb") as tile_file:
                            for chunk in tile_response.iter_content(1024):
                                tile_file.write(chunk)
                        
                        print(f"   ✅ Downloaded: {tile_filename}")
                        tiles.append((x, y, tile_path))
                        
                    else:
                        print(f"   ❌ Failed tile ({x},{y}): HTTP {tile_response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"   ❌ Network error downloading tile ({x},{y}): {e}")
                except Exception as e:
                    print(f"   ❌ Error downloading tile ({x},{y}): {e}")
        
        print(f"✅ Downloaded {len(tiles)}/{total_tiles} tiles successfully")
        return tiles

    # ---------------------------
    # Downloading / combining (updated for transactional support)
    # ---------------------------
    def download_tile(self, x: int, y: int, zoom: int, date: str, output_dir: str) -> Dict[str, Any]:
        url = f"{self.base_url}/{zoom}/{x}/{y}.jpg"
        
        # Handle date parameter - could be single date or date range
        if 'since=' in date and 'until=' in date:
            # Date range format: "since=2024-09-03&until=2025-09-03"
            # Parse the date range
            parts = date.split('&')
            since_date = parts[0].split('=')[1]
            until_date = parts[1].split('=')[1]
            params = {
                'apikey': self.api_key, 
                'since': since_date, 
                'until': until_date,
                'size': self.tile_size
            }
        else:
            # Single date format: "2024-08-15"
            params = {
                'apikey': self.api_key, 
                'since': date, 
                'until': date,
                'size': self.tile_size
            }
        
        tile_filename = f"tile_{zoom}_{x}_{y}.jpg"
        tile_path = os.path.join(output_dir, tile_filename)
        
        result = {
            'tile_path': tile_path,
            'actual_date': None,
            'survey_id': None,
            'capture_date': None
        }
        
        if os.path.exists(tile_path):
            result['tile_path'] = tile_path
            return result
            
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Extract actual date information from response headers
            actual_date = None
            survey_id = None
            capture_date = None
            
            # Check various header fields that might contain the actual date
            headers = response.headers
            if 'x-nearmap-survey-date' in headers:
                actual_date = headers['x-nearmap-survey-date']
            elif 'x-survey-date' in headers:
                actual_date = headers['x-survey-date']
            elif 'date' in headers:
                # This is usually the response date, not the imagery date, but worth logging
                response_date = headers['date']
                
            if 'x-nearmap-survey-id' in headers:
                survey_id = headers['x-nearmap-survey-id']
            elif 'x-survey-id' in headers:
                survey_id = headers['x-survey-id']
                
            if 'x-nearmap-capture-date' in headers:
                capture_date = headers['x-nearmap-capture-date']
            elif 'x-capture-date' in headers:
                capture_date = headers['x-capture-date']
            
            # Log the actual date if found (only for first tile to avoid spam)
            if not hasattr(self, 'first_tile_logged'):
                self.first_tile_logged = True
                    
                if actual_date:
                    print(f"📅 Nearmap actual imagery date: {actual_date}")
                if survey_id:
                    print(f"🆔 Survey ID: {survey_id}")
                if capture_date:
                    print(f"📸 Capture date: {capture_date}")
                    
                # Log all available headers for debugging (first tile only)
                nearmap_headers = {k: v for k, v in headers.items() if 'nearmap' in k.lower() or 'survey' in k.lower() or 'capture' in k.lower()}
                if nearmap_headers:
                    print(f"🔍 Nearmap response headers: {nearmap_headers}")
            
            result.update({
                'actual_date': actual_date,
                'survey_id': survey_id,
                'capture_date': capture_date
            })
            
            with open(tile_path, 'wb') as f:
                f.write(response.content)
            print(f"📥 Downloaded tile {x},{y} ({self.tile_size}x{self.tile_size})")
            return result
            
        except requests.RequestException as e:
            print(f"❌ Failed to download tile {x},{y}: {e}")
            raise

    def combine_tiles(self, tiles_set: Set[Tuple[int, int]], zoom: int, tiles_dir: str, output_path: str) -> Image.Image:
        """
        Combine individual tiles into a single large image.
        Improved version with better error handling and tile verification.
        Missing tiles are replaced with black tiles.
        """
        if not tiles_set:
            raise ValueError("No tiles to combine")
        
        # Calculate grid dimensions
        tile_xs = [t[0] for t in tiles_set]
        tile_ys = [t[1] for t in tiles_set]
        min_x, max_x = min(tile_xs), max(tile_xs)
        min_y, max_y = min(tile_ys), max(tile_ys)
        
        # Calculate final image dimensions
        grid_width = max_x - min_x + 1
        grid_height = max_y - min_y + 1
        final_width = grid_width * self.tile_size
        final_height = grid_height * self.tile_size
        
        print(f"🔗 Combining {len(tiles_set)} tiles...")
        print(f"   Grid size: {grid_width} x {grid_height}")
        print(f"   Final image size: {final_width} x {final_height}")
        
        # Check if image dimensions exceed PIL limits
        max_dimension = 65500  # PIL's maximum dimension
        total_pixels = final_width * final_height
        max_pixels = 1000000000  # 1 billion pixels - practical limit for very large images
        
        if final_width > max_dimension or final_height > max_dimension or total_pixels > max_pixels:
            print(f"⚠️  Warning: Image size ({final_width}x{final_height}, {total_pixels:,} pixels) exceeds practical limits")
            print(f"   PIL max dimension: {max_dimension}, practical pixel limit: {max_pixels:,}")
            print(f"   Consider reducing the area or using a lower resolution.")
            
            # Calculate scale factor needed
            dimension_scale = 1.0
            pixel_scale = 1.0
            
            if final_width > max_dimension or final_height > max_dimension:
                dimension_scale = min(max_dimension / final_width, max_dimension / final_height)
                
            if total_pixels > max_pixels:
                pixel_scale = math.sqrt(max_pixels / total_pixels)
            
            scale_factor = min(dimension_scale, pixel_scale)
            
            if scale_factor < 1.0:
                final_width = int(final_width * scale_factor)
                final_height = int(final_height * scale_factor)
                tile_size_scaled = int(self.tile_size * scale_factor)
                print(f"   Scaling down to: {final_width}x{final_height} (scale: {scale_factor:.3f})")
            else:
                tile_size_scaled = self.tile_size
        else:
            scale_factor = 1.0
            tile_size_scaled = self.tile_size
        
        # Create empty image to hold all tiles
        combined = Image.new('RGB', (final_width, final_height), (0, 0, 0))
        
        tiles_processed = 0
        missing_tiles = []
        
        # Create a black tile for missing tiles
        black_tile = Image.new('RGB', (tile_size_scaled, tile_size_scaled), (0, 0, 0))
        
        # Process each tile in the set
        for tile_x, tile_y in sorted(tiles_set):
            tile_filename = f"tile_{zoom}_{tile_x}_{tile_y}.jpg"
            tile_path = os.path.join(tiles_dir, tile_filename)
            
            # Calculate position in the combined image
            paste_x = (tile_x - min_x) * tile_size_scaled
            paste_y = (tile_y - min_y) * tile_size_scaled
            
            if os.path.exists(tile_path):
                try:
                    # Load tile image
                    tile_img = Image.open(tile_path)
                    
                    # Verify tile dimensions on first tile
                    if tiles_processed == 0:
                        print(f"   First tile dimensions: {tile_img.size} (expected: {self.tile_size}x{self.tile_size})")
                    
                    # Resize tile if necessary to match expected tile size
                    if tile_img.size != (tile_size_scaled, tile_size_scaled):
                        tile_img = tile_img.resize((tile_size_scaled, tile_size_scaled), Image.LANCZOS)
                    
                    # Paste tile into combined image
                    combined.paste(tile_img, (paste_x, paste_y))
                    tile_img.close()
                    tiles_processed += 1
                    
                    if tiles_processed % 100 == 0:
                        print(f"   Processed {tiles_processed} tiles...")
                        
                except Exception as e:
                    print(f"⚠️  Warning: Error processing tile {tile_filename}: {e}")
                    print(f"   Using black tile instead")
                    combined.paste(black_tile, (paste_x, paste_y))
                    missing_tiles.append(tile_filename)
            else:
                print(f"   Missing tile {tile_filename} - using black tile")
                combined.paste(black_tile, (paste_x, paste_y))
                missing_tiles.append(tile_filename)
        
        print(f"   Successfully processed {tiles_processed} tiles")
        
        # Report any missing tiles
        if missing_tiles:
            print(f"   Replaced {len(missing_tiles)} missing tiles with black tiles:")
            for tile in missing_tiles[:10]:  # Show first 10 missing tiles
                print(f"     - {tile}")
            if len(missing_tiles) > 10:
                print(f"     ... and {len(missing_tiles) - 10} more")
        
        # Save the combined image with error handling for large images
        try:
            combined.save(output_path, 'JPEG', quality=85)  # Reduced quality for large images
            print(f"✅ Saved combined image to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            # Try saving as PNG or with lower quality
            try:
                png_path = output_path.replace('.jpg', '.png')
                combined.save(png_path, 'PNG')
                print(f"✅ Saved as PNG instead: {png_path}")
                return combined
            except Exception as e2:
                print(f"❌ Also failed to save as PNG: {e2}")
                raise e
        
        return combined

    def create_metadata(self, bbox: Dict[str, float], tiles_set: Set[Tuple[int, int]], 
                        zoom: int, date: str, image_size: Tuple[int, int], 
                        output_json: str, avg_lat: float, actual_date: str = None, 
                        survey_ids: set = None, capture_dates: set = None) -> Dict[str, Any]:
        if not tiles_set:
            raise ValueError("No tiles provided for metadata.")
        tile_xs = [t[0] for t in tiles_set]
        tile_ys = [t[1] for t in tiles_set]
        min_x, max_x = min(tile_xs), max(tile_xs)
        min_y, max_y = min(tile_ys), max(tile_ys)
        # Compute bounds from tile edges
        min_lon, max_lat = self.tile_to_lonlat(min_x, min_y, zoom)  # top-left of min tile rect
        max_lon, min_lat = self.tile_to_lonlat(max_x + 1, max_y + 1, zoom)  # bottom-right
        
        # Calculate both Web Mercator theoretical and actual geographic resolution
        web_mercator_mpp = self.meters_per_pixel_at_lat(zoom, avg_lat)
        actual_geographic_mpp = self.calculate_actual_resolution(min_lon, max_lon, min_lat, max_lat, 
                                                                image_size[0], image_size[1])
        
        metadata = {
            "source": "Nearmap",
            "date_requested": date,
            "date_actual": actual_date if actual_date else date,
            "survey_ids": list(survey_ids) if survey_ids else [],
            "capture_dates": list(capture_dates) if capture_dates else [],
            "zoom_level": zoom,
            "resolution_description": "~3-inch spatial resolution",
            "target_resolution_meters_per_pixel": self.target_mpp,
            "web_mercator_resolution_meters_per_pixel": web_mercator_mpp,
            "actual_resolution_meters_per_pixel": actual_geographic_mpp,
            "tile_size": self.tile_size,
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "bounds": {"min_lon": min_lon, "max_lon": max_lon, "min_lat": min_lat, "max_lat": max_lat},
            "original_bbox": bbox,
            "tiles_used": {
                "count": len(tiles_set),
                "min_x": min_x, "max_x": max_x,
                "min_y": min_y, "max_y": max_y
            },
            "coordinate_system": "WGS84 (EPSG:4326)",
            "created_at": datetime.now().isoformat(),
            "note": "actual_resolution_meters_per_pixel is calculated from geographic bounds, web_mercator_resolution_meters_per_pixel is theoretical"
        }
        with open(output_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata to: {output_json}")
        print(f"   Requested date: {date}")
        if actual_date and actual_date != date:
            print(f"   Actual imagery date: {actual_date}")
        print(f"   Web Mercator resolution: {web_mercator_mpp:.6f} m/px")
        print(f"   Actual geographic resolution: {actual_geographic_mpp:.6f} m/px")
        return metadata

    # ---------------------------
    # Main workflow
    # ---------------------------
    def extract_region(self, kml_path: str, date: str = None, output_dir: str = None, location_name: str = None) -> Dict[str, str]:
        print(f"🌍 NEARMAP TRANSACTIONAL EXTRACTION")
        print("=" * 50)
        if not location_name:
            location_name = Path(kml_path).stem
        
        if not output_dir:
            output_dir = f"data/images/{location_name}/"
            
        os.makedirs(output_dir, exist_ok=True)
        tiles_dir = os.path.join(output_dir, "tiles")
        os.makedirs(tiles_dir, exist_ok=True)
        print(f"📍 Location: {location_name}")
        print(f"📁 Output: {output_dir}")

        # Step 1: Parse KML & compute bbox + polygon
        print("Step 1: Parsing KML boundary...")
        coordinates = self.parse_kml_boundary(kml_path)
        bbox = self.get_bounding_box(coordinates)
        polygon = self.create_polygon_from_coordinates(coordinates)
        self.boundary_polygon = polygon
        avg_lat = (bbox['min_lat'] + bbox['max_lat']) / 2.0
        avg_lon = (bbox['min_lon'] + bbox['max_lon']) / 2.0
        print(f"   Bounding box: {bbox['min_lon']:.6f}, {bbox['min_lat']:.6f} → {bbox['max_lon']:.6f}, {bbox['max_lat']:.6f}")

        # Step 2: Get transactional data (date, token, survey info)
        print("Step 2: Getting transactional coverage data...")
        
        if not date:
            # Use transactional method to get best date and transaction token
            final_date, transaction_token, survey_data = self.get_latest_imagery_date_and_transaction(coordinates)
        else:
            print(f"   Using specified date: {date}")
            # Try to get transaction token for the specified date
            final_date, transaction_token, survey_data = self.get_latest_imagery_date_and_transaction(coordinates)
            if survey_data:
                # Check if the survey matches the requested date
                survey_date = survey_data['captureDate'].split('T')[0]
                if survey_date != date:
                    print(f"   ⚠️ Requested date {date} not available, using closest: {survey_date}")
                    final_date = survey_date
                else:
                    final_date = date
            else:
                final_date = date
                transaction_token = None
        
        print(f"📅 Using imagery date: {final_date}")
        
        # Step 3: Try transactional method first
        if transaction_token and survey_data:
            print("Step 3: Using efficient transactional method...")
            return self._extract_with_transactional_method(
                survey_data, transaction_token, coordinates, bbox, 
                output_dir, location_name, final_date
            )
        else:
            print("Step 3: Falling back to tile-based method...")
            return self._extract_with_tile_method(
                coordinates, bbox, final_date, output_dir, location_name
            )
    
    def _extract_with_transactional_method(self, survey_data, transaction_token, coordinates, bbox, output_dir, location_name, final_date):
        """Extract using the efficient transactional method"""
        
        # Get survey tile grid information
        survey_id = survey_data['id']
        scale_info = survey_data.get('scale', {}).get('raster:Vert', {})
        width_tiles = scale_info.get('widthInTiles', 1)
        height_tiles = scale_info.get('heightInTiles', 1)
        
        print(f"   📊 Survey ID: {survey_id}")
        print(f"   📏 Grid size: {width_tiles}x{height_tiles} tiles")
        print(f"   🔑 Transaction token: {transaction_token[:50]}...")
        
        # Calculate estimated image size
        estimated_width = width_tiles * self.tile_size
        estimated_height = height_tiles * self.tile_size
        estimated_pixels = estimated_width * estimated_height
        
        print(f"   📐 Estimated final image: {estimated_width}x{estimated_height} ({estimated_pixels:,} pixels)")
        
        # Download tiles using transactional method
        tiles_dir = os.path.join(output_dir, "tiles")
        tiles = self.download_tiles_transactional(survey_id, transaction_token, width_tiles, height_tiles, tiles_dir)
        
        if not tiles:
            raise ValueError("No tiles downloaded successfully with transactional method")
        
        # Combine tiles
        print("🔧 Combining tiles...")
        if tiles:
            # Load first tile to get dimensions
            first_tile_img = Image.open(tiles[0][2])
            tile_width, tile_height = first_tile_img.size
            first_tile_img.close()
            
            # Create combined image
            total_width = width_tiles * tile_width
            total_height = height_tiles * tile_height
            combined_image = Image.new('RGB', (total_width, total_height))
            
            print(f"📐 Tile size: {tile_width}x{tile_height}")
            print(f"📐 Final size: {total_width}x{total_height}")
            
            # Paste each tile
            for x, y, tile_path in tiles:
                tile_img = Image.open(tile_path)
                paste_x = x * tile_width
                paste_y = y * tile_height
                combined_image.paste(tile_img, (paste_x, paste_y))
                tile_img.close()
            
            # Save final image
            image_path = os.path.join(output_dir, f"{location_name}_highres.jpg")
            combined_image.save(image_path, 'JPEG', quality=95)
            combined_image.close()
            
            print(f"✅ Combined image saved: {image_path}")
            
            # Create metadata
            print("💾 Creating metadata...")
            metadata_path = os.path.join(output_dir, f"{location_name}_highres_metadata.json")
            
            # Calculate actual resolution
            actual_resolution = self.calculate_actual_resolution(
                bbox['min_lon'], bbox['max_lon'], bbox['min_lat'], bbox['max_lat'],
                total_width, total_height
            )
            
            metadata = {
                "file": f"{location_name}_highres.jpg",
                "resolution_m_per_pixel": round(actual_resolution, 6),
                "spatial_reference": "EPSG:3857",
                "width_px": total_width,
                "height_px": total_height,
                "bounds": bbox,
                "extraction_method": "transactional",
                "survey_id": survey_data['id'],
                "capture_date": survey_data['captureDate'],
                "transaction_token_used": transaction_token[:20] + "...",
                "tile_grid": {
                    "width_tiles": width_tiles,
                    "height_tiles": height_tiles,
                    "tile_size": self.tile_size,
                    "total_tiles": len(tiles)
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"💾 Metadata saved: {metadata_path}")
            
            # Clean up temp files
            print("🧹 Cleaning up temporary files...")
            for x, y, tile_path in tiles:
                try:
                    os.remove(tile_path)
                except:
                    pass
            
            # Try to remove tiles directory if empty
            try:
                os.rmdir(tiles_dir)
            except:
                pass
            
            return {
                'image_path': image_path,
                'metadata_path': metadata_path,
                'location': location_name,
                'date_requested': final_date,
                'date_actual': survey_data['captureDate'].split('T')[0],
                'resolution_m_per_pixel': actual_resolution,
                'width_px': total_width,
                'height_px': total_height,
                'tiles_downloaded': len(tiles),
                'extraction_method': 'transactional',
                'survey_id': survey_data['id']
            }
        
        else:
            raise ValueError("Failed to download tiles with transactional method")
    
    def _extract_with_tile_method(self, coordinates, bbox, final_date, output_dir, location_name):
        """Fallback to the original tile-based method"""
        
        avg_lat = (bbox['min_lat'] + bbox['max_lat']) / 2.0
        polygon = self.create_polygon_from_coordinates(coordinates)
        
        # Pick zoom that achieves target resolution at this latitude
        zoom = self.zoom_for_target_mpp(self.target_mpp, avg_lat)
        actual_mpp = self.meters_per_pixel_at_lat(zoom, avg_lat)
        print(f"   Target m/px: {self.target_mpp}  |  Chosen zoom: {zoom}  |  Actual m/px: {actual_mpp:.4f}")

        # Determine intersecting tiles
        print("   Determining intersecting tiles...")
        intersecting_tiles = self.get_intersecting_tiles(bbox, zoom, polygon)
        tiles_needed = len(intersecting_tiles)
        if tiles_needed == 0:
            raise ValueError("No tiles intersect the provided KML polygon at the chosen zoom.")
        
        print(f"   Tiles to download: {tiles_needed} tiles of {self.tile_size}x{self.tile_size} pixels each")
        
        # Download tiles using original method
        print("   Downloading tiles...")
        tiles_dir = os.path.join(output_dir, "tiles")
        start_time = time.time()
        
        for (tile_x, tile_y) in sorted(intersecting_tiles):
            try:
                self.download_tile(tile_x, tile_y, zoom, final_date, tiles_dir)
                time.sleep(0.05)  # gentle rate limit
            except Exception as e:
                print(f"⚠️ Warning: Failed to download tile {tile_x},{tile_y}: {e}")
        
        download_time = time.time() - start_time
        print(f"   Downloaded in {download_time:.1f} seconds")
        
        # Combine tiles
        print("   Combining tiles...")
        image_path = os.path.join(output_dir, f"{location_name}_highres.jpg")
        combined_image = self.combine_tiles(intersecting_tiles, zoom, tiles_dir, image_path)

        # Create metadata
        print("   Creating metadata...")
        metadata_path = os.path.join(output_dir, f"{location_name}_highres_metadata.json")
        metadata = self.create_metadata(bbox, intersecting_tiles, zoom, final_date, combined_image.size, 
                                      metadata_path, avg_lat, final_date, set(), set())

        return {
            'image_path': image_path,
            'metadata_path': metadata_path,
            'location': location_name,
            'date_requested': final_date,
            'date_actual': final_date,
            'resolution_m_per_pixel': metadata.get('resolution_m_per_pixel', actual_mpp),
            'width_px': combined_image.size[0],
            'height_px': combined_image.size[1],
            'tiles_downloaded': tiles_needed,
            'extraction_method': 'tile-based',
            'zoom_level': zoom
        }


def main():
    """Main function with manual configuration variables - edit these to customize your extraction"""
    print("🚀 NEARMAP EXTRACTOR v4.0 - Manual Configuration")
    print("=" * 60)
    
    # === CONFIGURATION VARIABLES - EDIT THESE ===
    location = "Crossroads"  # Location name for output files
    kml_file = f"data/{location}/{location}.kml"  # Path to KML boundary file
    date = "2024-08-26"  # Date in YYYY-MM-DD format (set to None for auto-detect latest)
    output_dir = f"data/{location}/images"  # Output directory for image and metadata
    api_key = "ZTBkNjI1NDYtZmVhMy00MDA0LTk4NDUtZGNkYzY2MzBmNzg2"  # Nearmap API key
    target_mpp = 0.075  # Target meters per pixel (≈ 3 inches per pixel)
    # ==========================================
    
    print(f"📁 KML File: {kml_file}")
    print(f"📅 Date: {date if date else 'AUTO-DETECT LATEST'}")
    print(f"📂 Output Dir: {output_dir}")
    print(f"🏷️ Location: {location}")
    print(f"🎯 Target m/px: {target_mpp}")
    print()
    
    try:
        # Validate date format if provided
        if date:
            from datetime import datetime
            datetime.strptime(date, '%Y-%m-%d')
        
        # Check if KML file exists
        if not os.path.exists(kml_file):
            raise FileNotFoundError(f"KML file not found: {kml_file}")
        
        # Initialize extractor with optimal tile sizing
        extractor = NearmapTileExtractor(api_key=api_key, target_mpp=target_mpp)
        print(f"📐 Optimal tile size: {extractor.tile_size}x{extractor.tile_size} pixels")
        print()
        
        # Run extraction
        results = extractor.extract_region(
            kml_path=kml_file, 
            date=date,  # None for auto-detect, or specific date
            output_dir=output_dir, 
            location_name=location
        )
        
        print("\n📊 EXTRACTION RESULTS:")
        print(f"   📸 Image: {results['image']}")
        print(f"   📋 Metadata: {results['metadata']}")
        print(f"   📅 Date used: {results['date']}")
        print(f"   🎯 Achieved resolution: {results['actual_mpp']:.4f} m/px")
        print(f"   📐 Tile configuration: {results['tiles_downloaded']} tiles @ {results['tile_size']}x{results['tile_size']}px")
        print(f"   🔍 Zoom level: {results['zoom']}")
        
        return results
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return None
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        return None  
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

if __name__ == "__main__":
    print("🌍 NEARMAP TILE EXTRACTOR")
    print("⚙️ Using variable configuration...")
    print()
    
    results = main()
    
    if results:
        print(f"\n✅ SUCCESS! Files have been created and are ready to use.")
    else:
        print(f"\n❌ FAILED! Check the error messages above.")
    
    print(f"\n💡 To customize the extraction:")
    print(f"   Edit the variables in the main() function:")
    print(f"   • location: Change the location name")
    print(f"   • kml_file: Path to your KML boundary file")
    print(f"   • date: Date in YYYY-MM-DD format") 
    print(f"   • output_dir: Where to save files")
    print(f"   • api_key: Your Nearmap API key")
    print(f"   • target_mpp: Desired resolution (default 0.075 ≈ 3 in/px)")
