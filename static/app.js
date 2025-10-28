/**
 * ArborNote Tree Detection Interface - Main JavaScript
 * Production-ready implementation for interactive tree detection mapping
 */

// Production configuration
const PRODUCTION_CONFIG = {
    DEBUG_LOGGING: window.location.hostname === 'localhost' || window.location.search.includes('debug=true'),
    PERFORMANCE_MONITORING: true,
    ERROR_REPORTING: true,
    VERSION: '3.0.0'
};

// Enhanced logging function for production
function debugLog(message, data = null) {
    if (PRODUCTION_CONFIG.DEBUG_LOGGING) {
        if (data) {
            console.log(`[ArborNote Debug] ${message}`, data);
        } else {
            console.log(`[ArborNote Debug] ${message}`);
        }
    }
}

function errorLog(message, error = null) {
    console.error(`[ArborNote Error] ${message}`, error);
    // In production, you might want to send this to an error tracking service
    if (PRODUCTION_CONFIG.ERROR_REPORTING && typeof window.errorTracker !== 'undefined') {
        window.errorTracker.captureException(error || new Error(message));
    }
}

// Global application state
let map, drawnItems, currentGeofence, currentJob, socket;
let treePins = [];
let isDrawing = false;
let isEditing = false;

// ID mapping for new ArborNote-style interface
const ID_MAP = {
    'draw-polygon': 'clearMapBtn', // Repurpose as draw button
    'edit-polygon': 'clearMapBtn', 
    'add-tree': 'clearMapBtn',
    'remove-tree': 'clearMapBtn',
    'clear-map': 'clearMapBtn',
    'start-processing': 'startProcessingBtn',
    'kml-file-input': 'kmlUpload',
    'confidence-threshold': 'confidenceSlider',
    'iou-threshold': 'confidenceSlider',
    'download-all': 'downloadAll',
    'model-select': 'modelSelect',
    'imagery-date': 'dateSelect',
    'address-search': 'addressInput',
    'search-button': 'geocodeBtn',
    'search-results': 'statusMessage'
};

// Compatibility function to get elements by old or new ID
function getElement(oldId) {
    return document.getElementById(oldId) || document.getElementById(ID_MAP[oldId]);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    initializeUI();
    initializeWebSocket();
    loadAvailableData();
});

// ============================================================================
// MAP INITIALIZATION
// ============================================================================

function initializeMap() {
    // Initialize Leaflet map
    map = L.map('map', {
        center: [39.8283, -98.5795], // Center of USA
        zoom: 4,
        zoomControl: true
    });

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);

    // Initialize draw controls
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    // Initialize drawing controls
    const drawControl = new L.Control.Draw({
        position: 'topleft',
        draw: {
            polygon: {
                allowIntersection: false,
                drawError: {
                    color: '#e1e100',
                    message: '<strong>Error:</strong> Shape edges cannot cross!'
                },
                shapeOptions: {
                    color: '#3CB44B',
                    weight: 3,
                    opacity: 0.8,
                    fillOpacity: 0.2,
                    fillColor: '#3CB44B'
                }
            },
            polyline: false,
            rectangle: false,
            circle: false,
            marker: false,
            circlemarker: false
        },
        edit: {
            featureGroup: drawnItems,
            remove: true
        }
    });

    // Add draw control to map
    map.addControl(drawControl);

    // Map event listeners
    map.on('draw:created', onGeofenceCreated);
    map.on('draw:edited', onGeofenceEdited);
    map.on('draw:deleted', onGeofenceDeleted);
    map.on('draw:drawstart', onDrawStart);
    map.on('click', onMapClick);
}

// ============================================================================
// UI INITIALIZATION
// ============================================================================

function initializeUI() {
    // Initialize new ArborNote-style controls
    const clearMapBtn = document.getElementById('clearMapBtn');
    const startProcessingBtn = document.getElementById('startProcessingBtn');
    const geocodeBtn = document.getElementById('geocodeBtn');
    const addressInput = document.getElementById('addressInput');
    
    // Clear map functionality (combines old drawing tools)
    if (clearMapBtn) {
        clearMapBtn.addEventListener('click', clearGeofence);
    }

    // Processing button
    if (startProcessingBtn) {
        startProcessingBtn.addEventListener('click', startProcessing);
    }
    
    // Address search
    if (geocodeBtn && addressInput) {
        geocodeBtn.addEventListener('click', function() {
            const address = addressInput.value.trim();
            if (address) {
                searchAddress(address);
            }
        });
        
        addressInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                geocodeBtn.click();
            }
        });
    }
    
    // KML Upload functionality
    const kmlUpload = document.getElementById('kmlUpload');
    if (kmlUpload) {
        kmlUpload.addEventListener('change', handleKMLUpload);
        debugLog('KML upload event listener attached');
    }
    
    // Settings controls
    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceValue = document.getElementById('confidenceValue');
    if (confidenceSlider && confidenceValue) {
        confidenceSlider.addEventListener('input', function() {
            confidenceValue.textContent = this.value;
        });
        confidenceSlider.addEventListener('change', validateSettings);
    }
    
    // Download buttons with proper file type mapping
    const downloadButtons = {
        'downloadCsv': 'predictions_csv',
        'downloadKml': 'geofence_kml', 
        'downloadImage': 'visualization_image',
        'downloadOriginal': 'raw_image',
        'downloadProcessed': 'processed_image',
        'downloadMetadata': 'metadata',
        'downloadAll': 'all'
    };
    
    Object.entries(downloadButtons).forEach(([buttonId, fileType]) => {
        const button = document.getElementById(buttonId);
        if (button) {
            button.addEventListener('click', function() {
                debugLog(`Download button clicked: ${buttonId} -> ${fileType}`);
                if (buttonId === 'downloadAll') {
                    downloadAllFiles();
                } else {
                    downloadFile(fileType);
                }
            });
            console.log(`Download button initialized: ${buttonId} -> ${fileType}`);
        } else {
            console.warn(`Download button not found: ${buttonId}`);
        }
    });
    
    // Address search functionality
    initializeAddressSearch();
}

function initializeWebSocket() {
    socket = io();
    
    socket.on('job_progress', function(data) {
        updateProgress(data);
    });
    
    socket.on('connect', function() {
        console.log('Connected to server');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });
}

// ============================================================================
// DATA LOADING
// ============================================================================

async function loadAvailableData() {
    try {
        // Load available models
        const modelsResponse = await fetch('/api/models');
        const models = await modelsResponse.json();
        populateModelSelect(models);

        // Load available dates
        const datesResponse = await fetch('/api/dates');
        const dates = await datesResponse.json();
        populateDateSelect(dates);

    } catch (error) {
        console.error('Error loading data:', error);
        showAlert('Error loading application data', 'error');
    }
}

function populateModelSelect(models) {
    const select = document.getElementById('modelSelect');
    select.innerHTML = '';
    
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = `${model.name} ${!model.exists ? '(Missing)' : ''}`;
        option.disabled = !model.exists;
        if (model.default && model.exists) {
            option.selected = true;
        }
        select.appendChild(option);
    });
}

function populateDateSelect(dates) {
    const select = document.getElementById('imagery-date');
    // Keep the default "auto-detect" option
    
    dates.forEach(date => {
        const option = document.createElement('option');
        option.value = date.value;
        option.textContent = date.label;
        select.appendChild(option);
    });
}

// ============================================================================
// ADDRESS SEARCH FUNCTIONALITY
// ============================================================================

let searchTimeout = null;
let currentSearchMarker = null;

function initializeAddressSearch() {
    const searchInput = document.getElementById('address-search');
    const searchButton = document.getElementById('search-button');
    const searchResults = document.getElementById('search-results');
    
    // Search input event listeners
    searchInput.addEventListener('input', function() {
        const query = this.value.trim();
        
        // Clear previous timeout
        if (searchTimeout) {
            clearTimeout(searchTimeout);
        }
        
        if (query.length < 3) {
            hideSearchResults();
            return;
        }
        
        // Debounce search requests
        searchTimeout = setTimeout(() => {
            performAddressSearch(query);
        }, 300);
    });
    
    // Handle Enter key
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            const query = this.value.trim();
            if (query.length >= 3) {
                performAddressSearch(query);
            }
        }
        
        // Handle arrow key navigation
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            navigateSearchResults('down');
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            navigateSearchResults('up');
        } else if (e.key === 'Escape') {
            hideSearchResults();
            this.blur();
        }
    });
    
    // Search button click
    searchButton.addEventListener('click', function() {
        const query = searchInput.value.trim();
        if (query.length >= 3) {
            performAddressSearch(query);
        }
    });
    
    // Hide results when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.map-search')) {
            hideSearchResults();
        }
    });
}

async function performAddressSearch(query) {
    const searchResults = document.getElementById('search-results');
    
    try {
        // Show loading state
        showSearchResults();
        searchResults.innerHTML = '<div class="search-loading"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';
        
        // Make API request
        const response = await fetch(`/api/geocode?query=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Search failed');
        }
        
        displaySearchResults(data.results);
        
    } catch (error) {
        console.error('Search error:', error);
        searchResults.innerHTML = '<div class="search-no-results"><i class="fas fa-exclamation-triangle"></i> Search failed. Please try again.</div>';
    }
}

function displaySearchResults(results) {
    const searchResults = document.getElementById('search-results');
    
    if (results.length === 0) {
        searchResults.innerHTML = '<div class="search-no-results"><i class="fas fa-map-marker-alt"></i> No locations found</div>';
        return;
    }
    
    let html = '';
    results.forEach((result, index) => {
        const title = getLocationTitle(result);
        const subtitle = result.display_name;
        
        html += `
            <div class="search-result-item" data-index="${index}" data-lat="${result.lat}" data-lon="${result.lon}">
                <div class="search-result-title">${escapeHtml(title)}</div>
                <div class="search-result-subtitle">${escapeHtml(subtitle)}</div>
            </div>
        `;
    });
    
    searchResults.innerHTML = html;
    
    // Add click listeners
    searchResults.querySelectorAll('.search-result-item').forEach(item => {
        item.addEventListener('click', function() {
            const lat = parseFloat(this.dataset.lat);
            const lon = parseFloat(this.dataset.lon);
            const title = this.querySelector('.search-result-title').textContent;
            
            selectSearchResult(lat, lon, title);
        });
    });
}

function getLocationTitle(result) {
    // Extract a meaningful title from the address components
    const address = result.address || {};
    
    // Priority order for title
    const titleOptions = [
        address.house_number && address.road ? `${address.house_number} ${address.road}` : null,
        address.road,
        address.neighbourhood,
        address.suburb,
        address.city || address.town || address.village,
        address.county,
        address.state,
        result.display_name.split(',')[0]
    ];
    
    for (const option of titleOptions) {
        if (option && option.trim()) {
            return option.trim();
        }
    }
    
    return 'Location';
}

function selectSearchResult(lat, lon, title) {
    // Center map on selected location
    map.setView([lat, lon], 16);
    
    // Remove previous search marker
    if (currentSearchMarker) {
        map.removeLayer(currentSearchMarker);
    }
    
    // Add new marker
    currentSearchMarker = L.marker([lat, lon], {
        icon: L.divIcon({
            className: 'search-marker',
            html: '<i class="fas fa-map-marker-alt" style="color: #e74c3c; font-size: 20px;"></i>',
            iconSize: [20, 20],
            iconAnchor: [10, 20]
        })
    }).addTo(map);
    
    currentSearchMarker.bindPopup(`<strong>Search Result</strong><br>${escapeHtml(title)}`).openPopup();
    
    // Hide search results
    hideSearchResults();
    
    // Clear search input
    document.getElementById('address-search').value = title;
    
    console.log(`Navigated to: ${title} (${lat}, ${lon})`);
}

function navigateSearchResults(direction) {
    const items = document.querySelectorAll('.search-result-item');
    if (items.length === 0) return;
    
    const currentActive = document.querySelector('.search-result-item.active');
    let newIndex = 0;
    
    if (currentActive) {
        currentActive.classList.remove('active');
        const currentIndex = parseInt(currentActive.dataset.index);
        
        if (direction === 'down') {
            newIndex = currentIndex + 1;
            if (newIndex >= items.length) newIndex = 0;
        } else {
            newIndex = currentIndex - 1;
            if (newIndex < 0) newIndex = items.length - 1;
        }
    }
    
    items[newIndex].classList.add('active');
    items[newIndex].scrollIntoView({ block: 'nearest' });
}

function showSearchResults() {
    document.getElementById('search-results').style.display = 'block';
}

function hideSearchResults() {
    document.getElementById('search-results').style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// ZOOM UTILITY FUNCTIONS
// ============================================================================

function zoomToPolygon(polygon, delay = 300) {
    // Auto-zoom and center to the polygon
    setTimeout(() => {
        try {
            const bounds = polygon.getBounds();
            if (bounds.isValid()) {
                map.fitBounds(bounds.pad(0.2), {
                    maxZoom: 16,
                    animate: true,
                    duration: 1.0
                });
                console.log('Map auto-zoomed to polygon');
            } else {
                console.error('Invalid polygon bounds for zoom');
            }
        } catch (zoomError) {
            console.error('Error auto-zooming to polygon:', zoomError);
        }
    }, delay);
}

// ============================================================================
// MAP DRAWING FUNCTIONS
// ============================================================================

function startDrawing() {
    // Always check if there are existing polygons and confirm clearing
    if (currentGeofence || drawnItems.getLayers().length > 0) {
        if (!confirm('This will delete the current polygon and start fresh. Continue?')) {
            return;
        }
        // Clear all existing polygons
        clearGeofence();
    }
    
    // Trigger Leaflet draw
    document.querySelector('.leaflet-draw-draw-polygon').click();
    isDrawing = true;
    
    document.getElementById('draw-polygon').classList.add('active');
    showAlert('Click on the map to start drawing your geofence area', 'info');
}

function startEditing() {
    if (!currentGeofence) {
        showAlert('No geofence to edit. Draw one first.', 'warning');
        return;
    }
    
    // Trigger Leaflet edit
    document.querySelector('.leaflet-draw-edit-edit').click();
    isEditing = true;
    
    document.getElementById('edit-polygon').classList.add('active');
    showAlert('Click and drag the geofence points to modify the area', 'info');
}

function onDrawStart(e) {
    // If there's already a polygon drawn, show warning
    if (currentGeofence || drawnItems.getLayers().length > 0) {
        // Cancel the current drawing
        map._toolbars.draw._modes.polygon.handler.disable();
        
        // Show confirmation dialog matching the requested message
        if (confirm('This will delete the current polygon and start fresh. Continue?')) {
            // User confirmed, clear existing polygon and start fresh
            clearGeofence();
            clearTreePins();
            hideResults();
            
            // Hide progress and results sections
            const progressSection = document.getElementById('progressSection');
            const resultsSection = document.getElementById('resultsSection');
            if (progressSection) progressSection.classList.add('hidden');
            if (resultsSection) resultsSection.classList.add('hidden');
            
            // Update status message
            const statusMessage = document.getElementById('statusMessage');
            if (statusMessage) {
                statusMessage.innerHTML = '<i class="fas fa-info-circle"></i> Previous boundary deleted. Draw your new polygon.';
                statusMessage.className = 'status-message status-info';
            }
            
            // Re-enable drawing after a short delay
            setTimeout(() => {
                if (map._toolbars && map._toolbars.draw && map._toolbars.draw._modes.polygon) {
                    map._toolbars.draw._modes.polygon.handler.enable();
                }
            }, 100);
        } else {
            // User cancelled, don't start drawing
            console.log('User cancelled drawing new polygon');
        }
    }
}

function onGeofenceCreated(e) {
    const layer = e.layer;
    drawnItems.addLayer(layer);
    currentGeofence = layer;
    
    isDrawing = false;
    
    // Enable the start processing button
    const startBtn = document.getElementById('startProcessingBtn');
    if (startBtn) {
        startBtn.disabled = false;
    }
    
    // Auto-zoom and center to the newly drawn polygon
    zoomToPolygon(layer);
    
    // Calculate and display area information
    calculateAndDisplayArea();
    
    // Show success message
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.innerHTML = '<i class="fas fa-check-circle"></i> Polygon created! Ready to start tree detection.';
        statusMessage.className = 'status-message status-success';
    }
    
    console.log('Geofence created successfully');
}

function onGeofenceEdited(e) {
    isEditing = false;
    document.getElementById('edit-polygon').classList.remove('active');
    
    calculateAndDisplayArea();
    showAlert('Geofence updated!', 'success');
}

function onGeofenceDeleted(e) {
    currentGeofence = null;
    
    // Disable the start processing button
    const startBtn = document.getElementById('startProcessingBtn');
    if (startBtn) {
        startBtn.disabled = true;
    }
    
    // Hide area information
    hideAreaInfo();
    
    // Update status message
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.innerHTML = '<i class="fas fa-info-circle"></i> Draw a polygon to start analysis.';
        statusMessage.className = 'status-message status-info';
    }
    
    console.log('Geofence deleted');
}

async function calculateAndDisplayArea() {
    if (!currentGeofence) return;
    
    const coords = getGeofenceCoordinates();
    
    try {
        const response = await fetch('/api/calculate-area', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ geofence: coords })
        });
        
        const areaInfo = await response.json();
        displayAreaInfo(areaInfo);
        
    } catch (error) {
        console.error('Error calculating area:', error);
        showAlert('Error calculating area', 'error');
    }
}

function displayAreaInfo(areaInfo) {
    // Update new ArborNote-style area info display
    const areaAcres = document.getElementById('areaAcres');
    const estimatedCost = document.getElementById('estimatedCost');
    const estimatedTime = document.getElementById('estimatedTime');
    const areaInfoCard = document.getElementById('areaInfo');
    const startBtn = document.getElementById('startProcessingBtn');
    
    if (areaAcres) {
        areaAcres.textContent = `${areaInfo.acres} acres`;
    }
    
    if (estimatedCost) {
        estimatedCost.textContent = `$${areaInfo.cost_usd.toLocaleString()}`;
    }
    
    if (estimatedTime) {
        // Calculate estimated time using the correct formula: 0.1 * acres + 2
        const estimatedMinutes = Math.max(1, Math.round(areaInfo.acres * 0.1 + 2));
        estimatedTime.textContent = `${estimatedMinutes} min`;
    }
    
    // Show area info card and enable processing button
    if (areaInfoCard) {
        areaInfoCard.classList.remove('hidden');
    }
    
    if (startBtn) {
        startBtn.disabled = false;
    }
    
    console.log('Area info updated:', areaInfo);
}

function hideAreaInfo() {
    const areaInfoCard = document.getElementById('areaInfo');
    const startBtn = document.getElementById('startProcessingBtn');
    const areaAcres = document.getElementById('areaAcres');
    const estimatedCost = document.getElementById('estimatedCost');
    const estimatedTime = document.getElementById('estimatedTime');
    
    if (areaInfoCard) {
        areaInfoCard.classList.add('hidden');
    }
    
    if (startBtn) {
        startBtn.disabled = true;
    }
    
    // Reset values
    if (areaAcres) areaAcres.textContent = '-- acres';
    if (estimatedCost) estimatedCost.textContent = '$--';
    if (estimatedTime) estimatedTime.textContent = '-- min';
}

function getGeofenceCoordinates() {
    if (!currentGeofence) return [];
    
    const latLngs = currentGeofence.getLatLngs()[0];
    return latLngs.map(latlng => [latlng.lng, latlng.lat]);
}

// ============================================================================
// TREE PIN MANAGEMENT
// ============================================================================

function startAddingTrees() {
    const button = document.getElementById('add-tree');
    if (button.classList.contains('active')) {
        button.classList.remove('active');
        showAlert('Tree adding mode disabled', 'info');
    } else {
        button.classList.add('active');
        document.getElementById('remove-tree').classList.remove('active');
        showAlert('Click on the map to add tree pins', 'info');
    }
}

function startRemovingTrees() {
    const button = document.getElementById('remove-tree');
    if (button.classList.contains('active')) {
        button.classList.remove('active');
        showAlert('Tree removal mode disabled', 'info');
    } else {
        button.classList.add('active');
        document.getElementById('add-tree').classList.remove('active');
        showAlert('Click on tree pins to remove them', 'info');
    }
}

function onMapClick(e) {
    const addButton = document.getElementById('add-tree');
    const removeButton = document.getElementById('remove-tree');
    
    if (addButton.classList.contains('active')) {
        addTreePin(e.latlng);
    }
}

function addTreePin(latlng, confidence = null) {
    const confidenceBadge = confidence ? `<div class="confidence-badge">${Math.round(confidence * 100)}%</div>` : '';
    
    const treeIcon = L.divIcon({
        className: 'tree-pin-emoji',
        html: `<div style="font-size: 20px; text-align: center; line-height: 1;">🌲${confidenceBadge}</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
    });
    
    const marker = L.marker(latlng, { icon: treeIcon }).addTo(map);
    
    marker.on('click', function() {
        if (document.getElementById('remove-tree').classList.contains('active')) {
            removeTreePin(marker);
        } else {
            selectTreePin(marker);
        }
    });
    
    treePins.push({
        marker: marker,
        latlng: latlng,
        confidence: confidence,
        selected: false
    });
}

function removeTreePin(marker) {
    const index = treePins.findIndex(pin => pin.marker === marker);
    if (index !== -1) {
        map.removeLayer(marker);
        treePins.splice(index, 1);
    }
}

function selectTreePin(marker) {
    // Deselect all pins
    treePins.forEach(pin => {
        pin.selected = false;
        pin.marker.getElement().classList.remove('selected');
    });
    
    // Select clicked pin
    const pin = treePins.find(pin => pin.marker === marker);
    if (pin) {
        pin.selected = true;
        marker.getElement().classList.add('selected');
    }
}

function clearTreePins() {
    treePins.forEach(pin => {
        map.removeLayer(pin.marker);
    });
    treePins = [];
}

function displayTreePredictions(predictions) {
    // Clear any existing tree pins first
    clearTreePins();
    
    // Add a pin for each tree prediction
    predictions.forEach((prediction, index) => {
        const confidence = Math.round(prediction.confidence * 100);
        
        // Create custom icon based on confidence
        let iconColor = 'green';
        if (confidence < 50) iconColor = 'red';
        else if (confidence < 75) iconColor = 'orange';
        
        const treeIcon = L.divIcon({
            className: 'tree-pin-emoji',
            html: `<div style="font-size: 24px; text-align: center; line-height: 1; filter: ${confidence >= 75 ? 'hue-rotate(0deg)' : confidence >= 50 ? 'hue-rotate(30deg)' : 'hue-rotate(60deg)'};">🌲</div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12],
            popupAnchor: [0, -12]
        });
        
        const marker = L.marker([prediction.latitude, prediction.longitude], {
            icon: treeIcon
        });
        
        // Add popup with tree information
        marker.bindPopup(`
            <div class="tree-popup">
                <h4>Tree ${index + 1}</h4>
                <p><strong>Confidence:</strong> ${confidence}%</p>
                <p><strong>Location:</strong> ${prediction.latitude.toFixed(6)}, ${prediction.longitude.toFixed(6)}</p>
            </div>
        `);
        
        marker.addTo(map);
        
        // Store in treePins array
        treePins.push({
            marker: marker,
            lat: prediction.latitude,
            lng: prediction.longitude,
            confidence: prediction.confidence
        });
    });
    
    console.log(`Displayed ${predictions.length} tree predictions on map`);
}

// ============================================================================
// PROCESSING FUNCTIONS
// ============================================================================

async function startProcessing() {
    if (!currentGeofence) {
        showAlert('Please draw a geofence area first', 'warning');
        return;
    }
    
    console.log('Starting tree detection processing...');
    
    const coords = getGeofenceCoordinates();
    const settings = gatherSettings();
    
    const processingData = {
        geofence: coords,
        settings: settings,
        imagery_date: document.getElementById('dateSelect').value || '',
        model: document.getElementById('modelSelect').value || 'model_100'
    };
    
    console.log('Processing data:', processingData);
    
    try {
        // Disable the correct button ID
        const startBtn = document.getElementById('startProcessingBtn');
        if (startBtn) {
            startBtn.disabled = true;
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
        }
        
        // Show progress section
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.classList.remove('hidden');
        }
        
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(processingData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentJob = result.job_id;
            showAlert(`Processing started! Estimated time: ${result.estimated_time_minutes} minutes`, 'success');
            
            // Start polling for updates (backup for WebSocket)
            pollJobStatus();
            
        } else {
            throw new Error(result.error || 'Processing failed');
        }
        
    } catch (error) {
        console.error('Error starting processing:', error);
        showAlert(`Error: ${error.message}`, 'error');
        
        // Re-enable the correct button ID
        const startBtn = document.getElementById('startProcessingBtn');
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-play"></i> Start Tree Detection';
        }
        
        // Hide progress section on error
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.classList.add('hidden');
        }
    }
}

function gatherSettings() {
    // Get confidence threshold from the existing slider
    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceThreshold = confidenceSlider ? parseFloat(confidenceSlider.value) : 0.25;
    
    // Get project name from input field
    const projectNameInput = document.getElementById('projectName');
    const projectName = projectNameInput ? projectNameInput.value.trim() : '';
    
    return {
        confidence_threshold: confidenceThreshold,
        iou_threshold: 0.5, // Default value since this control doesn't exist yet
        patch_size: 512,    // Default value since this control doesn't exist yet
        enable_postprocessing: true, // Default value since this control doesn't exist yet
        project_name: projectName || null
    };
}

async function pollJobStatus() {
    if (!currentJob) return;
    
    try {
        const response = await fetch(`/api/job/${currentJob}`);
        const jobData = await response.json();
        
        updateProgress(jobData);
        
        if (jobData.status === 'processing' || jobData.status === 'queued') {
            setTimeout(pollJobStatus, 2000); // Poll every 2 seconds
        }
        
    } catch (error) {
        console.error('Error polling job status:', error);
    }
}

function updateProgress(data) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusMessage = document.getElementById('statusMessage');
    const progressSection = document.getElementById('progressSection');
    const startBtn = document.getElementById('startProcessingBtn');
    
    // Show progress section
    if (progressSection) {
        progressSection.classList.remove('hidden');
    }
    
    // Update progress bar
    if (progressFill) {
        progressFill.style.width = `${data.progress}%`;
    }
    
    if (progressText) {
        progressText.textContent = data.message || 'Processing...';
    }
    
    // Update status message
    if (statusMessage) {
        statusMessage.innerHTML = `<i class="fas fa-info-circle"></i> ${data.message || 'Processing...'}`;
        statusMessage.className = 'status-message status-info';
    }
    
    if (data.status === 'completed') {
        if (statusMessage) {
            statusMessage.innerHTML = `<i class="fas fa-check-circle"></i> Tree detection completed successfully!`;
            statusMessage.className = 'status-message status-success';
        }
        displayResults(data);
        if (startBtn) startBtn.disabled = false;
        
    } else if (data.status === 'error') {
        if (statusMessage) {
            statusMessage.innerHTML = `<i class="fas fa-exclamation-circle"></i> Processing failed: ${data.message}`;
            statusMessage.className = 'status-message status-error';
        }
        showProgress(false);
        if (startBtn) startBtn.disabled = false;
    }
}

function displayResults(jobData) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
    }
    
    // Ensure currentJob is set for downloads
    if (jobData.job_id && !currentJob) {
        currentJob = jobData.job_id;
        console.log(`Set currentJob for downloads: ${currentJob}`);
    }
    
    console.log('Displaying results:', jobData);
    console.log(`Current job for downloads: ${currentJob}`);
    
    // Update result statistics
    if (jobData.results) {
        const treeCount = document.getElementById('treeCount');
        const avgConfidence = document.getElementById('avgConfidence');
        
        if (treeCount) {
            treeCount.textContent = jobData.results.tree_count || 0;
        }
        
        if (avgConfidence) {
            avgConfidence.textContent = jobData.results.avg_confidence ? 
                `${Math.round(jobData.results.avg_confidence * 100)}%` : '0%';
        }
        
        // Display tree predictions on the map
        if (jobData.results.final_predictions && jobData.results.final_predictions.length > 0) {
            console.log(`Adding ${jobData.results.final_predictions.length} tree pins to map`);
            displayTreePredictions(jobData.results.final_predictions);
            
            // Zoom to fit the tree predictions
            if (jobData.results.final_predictions.length > 0) {
                const group = new L.featureGroup(treePins.map(pin => pin.marker));
                map.fitBounds(group.getBounds().pad(0.1));
            }
        } else {
            console.log('No tree predictions to display');
        }
    }
    
    // Update file list
    console.log('Available files:', jobData.files);
    updateFileList(jobData.files || {});
    
    // Enable download button
    document.getElementById('download-all').disabled = false;
}

function updateFileList(files) {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '';
    
    // File type to friendly name mapping
    const fileNames = {
        'predictions_csv': 'Tree Predictions (CSV)',
        'geofence_kml': 'Geofence Boundary (KML)',
        'results_summary': 'Processing Summary (JSON)',
        'metadata': 'Image Metadata (JSON)',
        'raw_image': 'Raw Imagery (JPG)',
        'processed_image': 'Processed Image (JPG)',
        'geofence_mask': 'Geofence Mask (PNG)',
        'visualization_image': 'Tree Detection Visualization (JPG)'
    };
    
    // File type to icon mapping
    const fileIcons = {
        'predictions_csv': 'fa-table',
        'geofence_kml': 'fa-map-marker-alt',
        'results_summary': 'fa-file-alt',
        'metadata': 'fa-info-circle',
        'raw_image': 'fa-image',
        'processed_image': 'fa-image',
        'geofence_mask': 'fa-mask',
        'visualization_image': 'fa-eye'
    };
    
    Object.entries(files).forEach(([fileType, filePath]) => {
        const li = document.createElement('li');
        li.className = 'file-item';
        
        const fileName = fileNames[fileType] || fileType.replace(/_/g, ' ').toUpperCase();
        const icon = fileIcons[fileType] || 'fa-file';
        
        li.innerHTML = `
            <span><i class="fas ${icon}"></i> ${fileName}</span>
            <button onclick="downloadFile('${fileType}')" class="btn btn-sm">
                <i class="fas fa-download"></i>
            </button>
        `;
        fileList.appendChild(li);
    });
}

async function downloadFile(fileType) {
    if (!currentJob) {
        showAlert('No completed analysis to download', 'warning');
        console.warn('Download attempted but no currentJob set');
        return;
    }
    
    console.log(`Downloading file: ${fileType} for job: ${currentJob}`);
    
    try {
        const downloadUrl = `/api/download/${currentJob}/${fileType}`;
        console.log(`Download URL: ${downloadUrl}`);
        
        const response = await fetch(downloadUrl);
        console.log(`Download response status: ${response.status}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const contentDisposition = response.headers.get('content-disposition');
            let filename = `${fileType}_${currentJob}`;
            
            // Try to get filename from response headers
            if (contentDisposition) {
                const matches = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showAlert(`Downloaded ${fileType} successfully!`, 'success');
            console.log(`Successfully downloaded: ${filename}`);
        } else {
            const errorText = await response.text();
            throw new Error(`Download failed: ${response.status} - ${errorText}`);
        }
    } catch (error) {
        console.error('Error downloading file:', error);
        showAlert(`Error downloading ${fileType}: ${error.message}`, 'error');
    }
}

async function downloadAllFiles() {
    if (!currentJob) {
        showAlert('No completed analysis to download', 'warning');
        console.warn('Download all attempted but no currentJob set');
        return;
    }
    
    console.log(`Downloading all files for job: ${currentJob}`);
    
    try {
        const downloadUrl = `/api/download-all/${currentJob}`;
        console.log(`Download all URL: ${downloadUrl}`);
        
        const response = await fetch(downloadUrl);
        console.log(`Download all response status: ${response.status}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `tree_detection_results_${currentJob}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showAlert('Downloaded all files successfully!', 'success');
            console.log(`Successfully downloaded all files as ZIP`);
        } else {
            const errorText = await response.text();
            throw new Error(`Download failed: ${response.status} - ${errorText}`);
        }
    } catch (error) {
        console.error('Error downloading files:', error);
        showAlert('Error downloading files', 'error');
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function clearMap() {
    if (currentGeofence && !confirm('This will delete all boundaries drawn on the map. Continue?')) {
        return;
    }
    
    clearGeofence();
    clearTreePins();
    hideResults();
    
    // Reset UI state for new ArborNote interface
    const startBtn = document.getElementById('startProcessingBtn');
    if (startBtn) {
        startBtn.disabled = true;
    }
    
    // Hide progress and results sections
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    if (progressSection) progressSection.classList.add('hidden');
    if (resultsSection) resultsSection.classList.add('hidden');
    
    // Update status message
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.innerHTML = '<i class="fas fa-info-circle"></i> Map cleared. Draw a new polygon to start analysis.';
        statusMessage.className = 'status-message status-info';
    }
    
    console.log('Map cleared - all boundaries deleted');
}

function clearGeofence() {
    // Remove all drawn layers (handles multiple polygons if any exist)
    drawnItems.clearLayers();
    currentGeofence = null;
    
    // Hide area information
    hideAreaInfo();
    
    // Disable start processing button
    const startBtn = document.getElementById('startProcessingBtn');
    if (startBtn) {
        startBtn.disabled = true;
    }
    
    // Reset progress and results
    hideResults();
}

function hideResults() {
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('download-all').disabled = true;
    showProgress(false);
    currentJob = null;
}

function showProgress(show) {
    const container = document.getElementById('progress-container');
    if (show) {
        container.classList.remove('hidden');
    } else {
        container.classList.add('hidden');
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-text').textContent = 'Ready to start...';
    }
}

function showAlert(message, type = 'info') {
    // Get or create alerts container
    let alertsContainer = document.getElementById('alerts');
    if (!alertsContainer) {
        alertsContainer = document.createElement('div');
        alertsContainer.id = 'alerts';
        alertsContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 400px;
        `;
        document.body.appendChild(alertsContainer);
    }
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.style.cssText = `
        background: ${type === 'error' ? '#f8d7da' : type === 'success' ? '#d4edda' : type === 'warning' ? '#fff3cd' : '#d1ecf1'};
        color: ${type === 'error' ? '#721c24' : type === 'success' ? '#155724' : type === 'warning' ? '#856404' : '#0c5460'};
        border: 1px solid ${type === 'error' ? '#f5c6cb' : type === 'success' ? '#c3e6cb' : type === 'warning' ? '#ffeaa7' : '#bee5eb'};
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 14px;
    `;
    alert.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 1.2em; cursor: pointer; color: inherit;">&times;</button>
        </div>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-remove after 5 seconds for info/success messages
    if (type === 'info' || type === 'success') {
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }
}

function toggleAdvanced() {
    const expandable = document.querySelector('.expandable');
    const content = document.getElementById('advanced-settings');
    
    expandable.classList.toggle('expanded');
}

function validateSettings() {
    // Validate confidence threshold
    const confidence = parseFloat(document.getElementById('confidence-threshold').value);
    if (confidence < 0.1 || confidence > 1.0) {
        showAlert('Confidence threshold must be between 0.1 and 1.0', 'warning');
    }
    
    // Validate IOU threshold
    const iou = parseFloat(document.getElementById('iou-threshold').value);
    if (iou < 0.1 || iou > 1.0) {
        showAlert('IOU threshold must be between 0.1 and 1.0', 'warning');
    }
}

// ============================================================================
// KML UPLOAD FUNCTIONALITY
// ============================================================================

async function handleKMLUpload(event) {
    console.log('handleKMLUpload function called');
    const file = event.target.files[0];
    if (!file) {
        console.log('No file selected');
        return;
    }
    
    console.log('File selected:', file.name, 'Type:', file.type, 'Size:', file.size);
    
    // Check file type
    const validTypes = ['application/vnd.google-earth.kml+xml', 'application/vnd.google-earth.kmz', 'text/xml', 'application/xml'];
    const validExtensions = ['.kml', '.kmz'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    console.log('File extension:', fileExtension);
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        console.log('Invalid file type');
        showAlert('Please select a valid KML or KMZ file', 'error');
        return;
    }
    
    // Show loading status using the main status message
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.innerHTML = '<i class="fas fa-upload"></i> Uploading and processing KML file...';
        statusMessage.className = 'status-message status-info';
    }
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        console.log('Uploading KML file:', file.name, 'Size:', file.size, 'bytes');
        
        const response = await fetch('/api/upload-kml', {
            method: 'POST',
            body: formData
        });
        
        console.log('Upload response status:', response.status);
        
        const result = await response.json();
        console.log('Upload result:', result);
        
        if (result.success) {
            // Clear existing polygons
            drawnItems.clearLayers();
            currentGeofence = null;
            
            // Add the polygon to the map
            if (result.coordinates && result.coordinates.length > 0) {
                console.log('Creating polygon with coordinates:', result.coordinates.length, 'points');
                console.log('First few coordinates:', result.coordinates.slice(0, 3));
                
                // Backend already sends coordinates in correct [lat, lng] format
                let processedCoords = result.coordinates.map(coord => {
                    // Handle different coordinate formats
                    if (Array.isArray(coord) && coord.length >= 2) {
                        // Backend already converted to [lat, lng] format - use as-is
                        return [parseFloat(coord[0]), parseFloat(coord[1])];
                    } else if (typeof coord === 'object' && coord.lat && coord.lng) {
                        // Object format {lat: x, lng: y}
                        return [parseFloat(coord.lat), parseFloat(coord.lng)];
                    } else {
                        console.error('Invalid coordinate format:', coord);
                        return null;
                    }
                }).filter(coord => coord !== null);
                
                console.log('Processed coordinates:', processedCoords.length, 'valid points');
                console.log('Coordinate sample:', processedCoords.slice(0, 3));
                
                if (processedCoords.length < 3) {
                    throw new Error('Not enough valid coordinates to create a polygon');
                }
                
                const polygon = L.polygon(processedCoords, {
                    color: '#3CB44B',
                    fillColor: '#3CB44B',
                    fillOpacity: 0.2,
                    weight: 3,
                    opacity: 0.8
                });
                
                // Add polygon to map
                drawnItems.addLayer(polygon);
                currentGeofence = polygon;
                
                console.log('Polygon created and added to map');
                console.log('Polygon bounds:', polygon.getBounds());
                
                // Treat KML import exactly like a drawn polygon
                // Enable the start processing button (using correct ArborNote ID)
                const startBtn = document.getElementById('startProcessingBtn');
                if (startBtn) {
                    startBtn.disabled = false;
                }
                
                // Show the progress section (it's needed for status messages)
                const progressSection = document.getElementById('progressSection');
                if (progressSection) {
                    progressSection.classList.remove('hidden');
                }
                
                // Calculate and display area information (same as drawn polygons)
                calculateAndDisplayArea();
                
                // Show success message (same as drawn polygons)
                const statusMessage = document.getElementById('statusMessage');
                if (statusMessage) {
                    statusMessage.innerHTML = '<i class="fas fa-check-circle"></i> KML boundary imported! Ready to start tree detection.';
                    statusMessage.className = 'status-message status-success';
                }
                
                // Auto-zoom and center to the imported KML polygon
                zoomToPolygon(polygon);
                console.log('KML polygon treated as drawn boundary - ready for processing');
                
                // Show success alert
                showAlert(`KML file loaded successfully! Boundary created with ${processedCoords.length} points.`, 'success');
                
            } else {
                throw new Error('No valid coordinates found in the file');
            }
            
        } else {
            console.error('Upload failed:', result.error);
            throw new Error(result.error || 'Failed to process KML file');
        }
        
    } catch (error) {
        console.error('KML upload error:', error);
        
        // Update status message to show error
        const statusMessage = document.getElementById('statusMessage');
        if (statusMessage) {
            statusMessage.innerHTML = `<i class="fas fa-exclamation-triangle"></i> KML upload failed: ${error.message}`;
            statusMessage.className = 'status-message status-error';
        }
        
        // Also show a prominent alert
        showAlert(`KML upload failed: ${error.message}`, 'error');
    }
    
    // Clear the file input
    event.target.value = '';
}

// ============================================================================
// AREA INFORMATION UPDATE FROM KML
// ============================================================================

function updateAreaInfoFromKML(areaInfo) {
    console.log('Updating area info from KML:', areaInfo);
    
    if (!areaInfo) {
        console.warn('No area info provided');
        return;
    }
    
    // Show the area info section
    const areaSection = document.getElementById('area-info');
    if (areaSection) {
        areaSection.classList.remove('hidden');
    }
    
    // Update area display
    const areaAcres = document.getElementById('area-acres');
    const areaMeters = document.getElementById('area-meters');
    const totalCost = document.getElementById('total-cost');
    const estimatedTime = document.getElementById('estimated-time');
    
    if (areaAcres && areaInfo.acres !== undefined) {
        areaAcres.textContent = `${areaInfo.acres.toFixed(2)} acres`;
    }
    
    if (areaMeters && areaInfo.square_meters !== undefined) {
        areaMeters.textContent = `${areaInfo.square_meters.toLocaleString()} m²`;
    }
    
    if (totalCost && areaInfo.cost_usd !== undefined) {
        totalCost.textContent = areaInfo.cost_usd.toFixed(0);
    }
    
    // Calculate and update estimated processing time using correct formula: 0.1 * acres + 2
    if (estimatedTime && areaInfo.acres !== undefined) {
        const estimatedMinutes = Math.max(1, Math.round(areaInfo.acres * 0.1 + 2));
        estimatedTime.textContent = `${estimatedMinutes} minutes`;
    }
    
    // Enable the start processing button
    const startButton = document.getElementById('start-processing');
    if (startButton) {
        startButton.disabled = false;
        startButton.textContent = 'Start Tree Detection';
        console.log('Start processing button enabled');
    }
    
    // Update header stats if elements exist
    const headerArea = document.querySelector('.header-stats .stat:first-child .stat-value');
    const headerCost = document.querySelector('.header-stats .stat:last-child .stat-value');
    
    if (headerArea && areaInfo.acres !== undefined) {
        headerArea.textContent = `${areaInfo.acres.toFixed(1)} acres`;
    }
    
    if (headerCost && areaInfo.cost_usd !== undefined) {
        headerCost.textContent = `$${areaInfo.cost_usd.toFixed(0)}`;
    }
    
    console.log('Area info UI updated successfully');
}

// ============================================================================
// EXPORT FUNCTIONS FOR DEBUGGING
// ============================================================================

window.ArborNote = {
    map,
    treePins,
    currentGeofence,
    currentJob,
    clearMap,
    clearGeofence,
    startProcessing
};
