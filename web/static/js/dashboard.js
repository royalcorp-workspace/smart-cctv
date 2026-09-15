/**
 * Smart CCTV 2.0 - Master Client Orchestrator
 * Coordinates modular scripts: theme, camera_manager, telemetry, calibration, and incidents.
 */

let pollTimer = null;

// Digital clock update interval
setInterval(updateClock, 1000);
updateClock();

/* ==========================================================================
   LIFECYCLE INITIALIZATION
   ========================================================================== */

window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadCameraList();
  pollTelemetry();
  loadRecentEvents();

  pollTimer = setInterval(pollTelemetry, 1000);
  setInterval(loadCameraList, 8000);
  setInterval(loadRecentEvents, 10000);
});

/* ==========================================================================
   GLOBAL WINDOW EXPORTS (Guarantee inline HTML event resolution)
   ========================================================================== */

// Theme & Display
window.applyTheme = applyTheme;
window.toggleTheme = toggleTheme;
window.updateClock = updateClock;
window.toggleFullscreen = toggleFullscreen;
window.showToast = showToast;

// Camera & Telemetry
window.onCameraChange = onCameraChange;
window.loadCameraList = loadCameraList;
window.pollTelemetry = pollTelemetry;
window.onStreamLoad = onStreamLoad;
window.onStreamError = onStreamError;

// Camera Onboarding & Deletion
window.openAddCameraModal = openAddCameraModal;
window.closeAddCameraModal = closeAddCameraModal;
window.onCamNameInput = onCamNameInput;
window.updateRtspPreview = updateRtspPreview;
window.testRtspConnection = testRtspConnection;
window.submitAddCamera = submitAddCamera;
window.openDeleteCameraModal = openDeleteCameraModal;
window.closeDeleteCameraModal = closeDeleteCameraModal;
window.confirmDeleteCamera = confirmDeleteCamera;

// Calibration Canvas & Category Switching
window.toggleZoneEditor = toggleZoneEditor;
window.switchEditorCategory = switchEditorCategory;
window.onZoneSelectChange = onZoneSelectChange;
window.setZoneMode = setZoneMode;
window.renderZoneCanvas = renderZoneCanvas;
window.syncCanvasWithVideoFeed = syncCanvasWithVideoFeed;

// Calibration Drawer & Elements
window.togglePropertiesDrawer = togglePropertiesDrawer;
window.updateCurrentElemMeta = updateCurrentElemMeta;
window.openAddElementPrompt = openAddElementPrompt;
window.closeAddElementModal = closeAddElementModal;
window.submitAddElement = submitAddElement;
window.confirmAddElement = confirmAddElement;
window.deleteCurrentElement = deleteCurrentElement;
window.resetCurrentZone = resetCurrentZone;
window.deleteSelectedPoint = deleteSelectedPoint;
window.undoZoneAction = undoZoneAction;
window.saveZonesToServer = saveZonesToServer;
window.populateElementSelector = populateElementSelector;
window.openCalibrationGuide = openCalibrationGuide;
window.closeCalibrationGuide = closeCalibrationGuide;
window.promptDeleteZoneConfirm = promptDeleteZoneConfirm;
window.closeDeleteZoneConfirmModal = closeDeleteZoneConfirmModal;
window.confirmDeleteZoneAction = confirmDeleteZoneAction;

// Incident DVR & Video Modal
window.loadRecentEvents = loadRecentEvents;
window.openVideoPlayer = openVideoPlayer;
window.closeVideoModal = closeVideoModal;
