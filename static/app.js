/**
 * ArborNote Tree Detection Interface - Main JavaScript
 * MVP implementation for interactive tree detection mapping
 */

// Global application state
let map, drawnItems, currentGeofence, currentJob, socket;
let treePins = [];
let isDrawing = false;
let isEditing = false;

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
                    color: '#2e8b57',
                    weight: 3,
                    opacity: 0.8,
                    fillOpacity: 0.2
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

    // Add draw control to map but hide it (we'll use custom buttons)
    map.addControl(drawControl);
    document.querySelector('.leaflet-draw').style.display = 'none';

    // Map event listeners
    map.on('draw:created', onGeofenceCreated);
    map.on('draw:edited', onGeofenceEdited);
    map.on('draw:deleted', onGeofenceDeleted);
    map.on('click', onMapClick);
}

// ============================================================================
// UI INITIALIZATION
// ============================================================================

function initializeUI() {
    // Map control buttons
    document.getElementById('draw-polygon').addEventListener('click', startDrawing);
    document.getElementById('edit-polygon').addEventListener('click', startEditing);
    document.getElementById('add-tree').addEventListener('click', startAddingTrees);
    document.getElementById('remove-tree').addEventListener('click', startRemovingTrees);
    document.getElementById('clear-map').addEventListener('click', clearMap);

    // Processing button
    document.getElementById('start-processing').addEventListener('click', startProcessing);
    
    // Settings listeners
    document.getElementById('confidence-threshold').addEventListener('change', validateSettings);
    document.getElementById('iou-threshold').addEventListener('change', validateSettings);
    
    // Download button
    document.getElementById('download-all').addEventListener('click', downloadAllFiles);
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
    const select = document.getElementById('model-select');
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
// MAP DRAWING FUNCTIONS
// ============================================================================

function startDrawing() {
    if (currentGeofence) {
        if (!confirm('This will replace your current geofence. Continue?')) {
            return;
        }
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

function onGeofenceCreated(e) {
    const layer = e.layer;
    drawnItems.addLayer(layer);
    currentGeofence = layer;
    
    isDrawing = false;
    document.getElementById('draw-polygon').classList.remove('active');
    document.getElementById('edit-polygon').disabled = false;
    document.getElementById('start-processing').disabled = false;
    
    calculateAndDisplayArea();
    showAlert('Geofence created! You can now start processing or edit the area.', 'success');
}

function onGeofenceEdited(e) {
    isEditing = false;
    document.getElementById('edit-polygon').classList.remove('active');
    
    calculateAndDisplayArea();
    showAlert('Geofence updated!', 'success');
}

function onGeofenceDeleted(e) {
    currentGeofence = null;
    document.getElementById('edit-polygon').disabled = true;
    document.getElementById('start-processing').disabled = true;
    
    hideAreaInfo();
    showAlert('Geofence deleted', 'info');
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
    document.getElementById('area-acres').textContent = `${areaInfo.acres} acres`;
    document.getElementById('area-meters').textContent = `${areaInfo.square_meters.toLocaleString()} m²`;
    document.getElementById('total-cost').textContent = areaInfo.cost_usd.toLocaleString();
    
    // Update header display
    document.getElementById('cost-display').textContent = `Estimated Cost: $${areaInfo.cost_usd.toLocaleString()}`;
    document.getElementById('area-display').textContent = `Area: ${areaInfo.acres} acres`;
    
    // Show area info section
    document.getElementById('area-info').classList.remove('hidden');
}

function hideAreaInfo() {
    document.getElementById('area-info').classList.add('hidden');
    document.getElementById('cost-display').textContent = 'Estimated Cost: $0';
    document.getElementById('area-display').textContent = 'Area: 0 acres';
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
    const treeIcon = L.divIcon({
        className: 'tree-pin',
        html: confidence ? `<div class="confidence-badge">${Math.round(confidence * 100)}%</div>` : '',
        iconSize: [12, 12],
        iconAnchor: [6, 6]
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
            className: 'tree-pin',
            html: `<div class="tree-marker" style="background-color: ${iconColor};">
                     <i class="fas fa-tree"></i>
                   </div>`,
            iconSize: [20, 20],
            iconAnchor: [10, 10],
            popupAnchor: [0, -10]
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
    
    const coords = getGeofenceCoordinates();
    const settings = gatherSettings();
    
    const processingData = {
        geofence: coords,
        settings: settings,
        imagery_date: document.getElementById('imagery-date').value,
        model: document.getElementById('model-select').value
    };
    
    try {
        document.getElementById('start-processing').disabled = true;
        showProgress(true);
        
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
        document.getElementById('start-processing').disabled = false;
        showProgress(false);
    }
}

function gatherSettings() {
    return {
        confidence_threshold: parseFloat(document.getElementById('confidence-threshold').value),
        iou_threshold: parseFloat(document.getElementById('iou-threshold').value),
        patch_size: parseInt(document.getElementById('patch-size').value),
        enable_postprocessing: document.getElementById('enable-postprocessing').checked
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
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    progressFill.style.width = `${data.progress}%`;
    progressText.textContent = data.message || 'Processing...';
    
    if (data.status === 'completed') {
        showAlert('Tree detection completed successfully!', 'success');
        displayResults(data);
        document.getElementById('start-processing').disabled = false;
        
    } else if (data.status === 'error') {
        showAlert(`Processing failed: ${data.message}`, 'error');
        showProgress(false);
        document.getElementById('start-processing').disabled = false;
    }
}

function displayResults(jobData) {
    // Show results section
    document.getElementById('results-section').classList.remove('hidden');
    
    console.log('Displaying results:', jobData);
    
    // Update result statistics
    if (jobData.results) {
        document.getElementById('tree-count').textContent = jobData.results.tree_count || 0;
        document.getElementById('avg-confidence').textContent = 
            jobData.results.avg_confidence ? `${Math.round(jobData.results.avg_confidence * 100)}%` : '0%';
        
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
    if (!currentJob) return;
    
    try {
        const response = await fetch(`/api/download/${currentJob}/${fileType}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${fileType}_${currentJob}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
    } catch (error) {
        console.error('Error downloading file:', error);
        showAlert('Error downloading file', 'error');
    }
}

async function downloadAllFiles() {
    if (!currentJob) return;
    
    try {
        const response = await fetch(`/api/download-all/${currentJob}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `tree_detection_${currentJob}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
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
    if (currentGeofence && !confirm('This will clear all data on the map. Continue?')) {
        return;
    }
    
    clearGeofence();
    clearTreePins();
    hideResults();
    
    // Reset UI state
    document.getElementById('edit-polygon').disabled = true;
    document.getElementById('start-processing').disabled = true;
    document.getElementById('add-tree').disabled = false;
    document.getElementById('remove-tree').disabled = false;
    
    showAlert('Map cleared', 'info');
}

function clearGeofence() {
    if (currentGeofence) {
        drawnItems.removeLayer(currentGeofence);
        currentGeofence = null;
    }
    hideAreaInfo();
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
    const alertsContainer = document.getElementById('alerts');
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 1.2em; cursor: pointer;">&times;</button>
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
// EXPORT FUNCTIONS FOR DEBUGGING
// ============================================================================

window.ArborNote = {
    map,
    treePins,
    currentGeofence,
    currentJob,
    clearMap,
    startProcessing
};
