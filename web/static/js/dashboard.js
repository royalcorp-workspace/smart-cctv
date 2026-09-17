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
  if (typeof initGridView === "function") {
    initGridView();
  }
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

// Camera Onboarding, Deletion & Calibration (Admin Console Only)
if (window.IS_ADMIN) {
  // Camera Onboarding & Deletion
  window.openAddCameraModal = typeof openAddCameraModal !== "undefined" ? openAddCameraModal : null;
  window.closeAddCameraModal = typeof closeAddCameraModal !== "undefined" ? closeAddCameraModal : null;
  window.onCamNameInput = typeof onCamNameInput !== "undefined" ? onCamNameInput : null;
  window.updateRtspPreview = typeof updateRtspPreview !== "undefined" ? updateRtspPreview : null;
  window.testRtspConnection = typeof testRtspConnection !== "undefined" ? testRtspConnection : null;
  window.submitAddCamera = typeof submitAddCamera !== "undefined" ? submitAddCamera : null;
  window.openDeleteCameraModal = typeof openDeleteCameraModal !== "undefined" ? openDeleteCameraModal : null;
  window.closeDeleteCameraModal = typeof closeDeleteCameraModal !== "undefined" ? closeDeleteCameraModal : null;
  window.confirmDeleteCamera = typeof confirmDeleteCamera !== "undefined" ? confirmDeleteCamera : null;

  // Calibration Canvas & Category Switching
  window.toggleZoneEditor = typeof toggleZoneEditor !== "undefined" ? toggleZoneEditor : null;
  window.switchEditorCategory = typeof switchEditorCategory !== "undefined" ? switchEditorCategory : null;
  window.onZoneSelectChange = typeof onZoneSelectChange !== "undefined" ? onZoneSelectChange : null;
  window.setZoneMode = typeof setZoneMode !== "undefined" ? setZoneMode : null;
  window.renderZoneCanvas = typeof renderZoneCanvas !== "undefined" ? renderZoneCanvas : null;
  window.syncCanvasWithVideoFeed = typeof syncCanvasWithVideoFeed !== "undefined" ? syncCanvasWithVideoFeed : null;

  // Calibration Drawer & Elements
  window.togglePropertiesDrawer = typeof togglePropertiesDrawer !== "undefined" ? togglePropertiesDrawer : null;
  window.updateCurrentElemMeta = typeof updateCurrentElemMeta !== "undefined" ? updateCurrentElemMeta : null;
  window.openAddElementPrompt = typeof openAddElementPrompt !== "undefined" ? openAddElementPrompt : null;
  window.closeAddElementModal = typeof closeAddElementModal !== "undefined" ? closeAddElementModal : null;
  window.submitAddElement = typeof submitAddElement !== "undefined" ? submitAddElement : null;
  window.confirmAddElement = typeof confirmAddElement !== "undefined" ? confirmAddElement : null;
  window.deleteCurrentElement = typeof deleteCurrentElement !== "undefined" ? deleteCurrentElement : null;
  window.resetCurrentZone = typeof resetCurrentZone !== "undefined" ? resetCurrentZone : null;
  window.deleteSelectedPoint = typeof deleteSelectedPoint !== "undefined" ? deleteSelectedPoint : null;
  window.undoZoneAction = typeof undoZoneAction !== "undefined" ? undoZoneAction : null;
  window.saveZonesToServer = typeof saveZonesToServer !== "undefined" ? saveZonesToServer : null;
  window.populateElementSelector = typeof populateElementSelector !== "undefined" ? populateElementSelector : null;
  window.openCalibrationGuide = typeof openCalibrationGuide !== "undefined" ? openCalibrationGuide : null;
  window.closeCalibrationGuide = typeof closeCalibrationGuide !== "undefined" ? closeCalibrationGuide : null;
  window.promptDeleteZoneConfirm = typeof promptDeleteZoneConfirm !== "undefined" ? promptDeleteZoneConfirm : null;
  window.closeDeleteZoneConfirmModal = typeof closeDeleteZoneConfirmModal !== "undefined" ? closeDeleteZoneConfirmModal : null;
  window.confirmDeleteZoneAction = typeof confirmDeleteZoneAction !== "undefined" ? confirmDeleteZoneAction : null;
}

// Incident DVR & Video Modal
window.loadRecentEvents = loadRecentEvents;
window.openVideoPlayer = openVideoPlayer;
window.closeVideoModal = closeVideoModal;

// Grid View Matrix Orchestrator
window.switchViewMode = typeof switchViewMode !== "undefined" ? switchViewMode : null;
window.onGridSizeChange = typeof onGridSizeChange !== "undefined" ? onGridSizeChange : null;
window.focusCameraSingleView = typeof focusCameraSingleView !== "undefined" ? focusCameraSingleView : null;
window.renderGridMatrix = typeof renderGridMatrix !== "undefined" ? renderGridMatrix : null;
window.reloadGridCameras = typeof reloadGridCameras !== "undefined" ? reloadGridCameras : null;
window.prevGridPage = typeof prevGridPage !== "undefined" ? prevGridPage : null;
window.nextGridPage = typeof nextGridPage !== "undefined" ? nextGridPage : null;
window.toggleAutoTour = typeof toggleAutoTour !== "undefined" ? toggleAutoTour : null;
window.onTourIntervalChange = typeof onTourIntervalChange !== "undefined" ? onTourIntervalChange : null;
