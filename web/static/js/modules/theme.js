/**
 * Smart CCTV 2.0 - Theme Switcher, Clock, & Fullscreen Module
 */

const THEME_STORAGE_KEY = "smart_cctv_theme";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const toggleBtn = document.getElementById("themeToggle");
  if (toggleBtn) {
    const isDark = theme === "dark";
    toggleBtn.setAttribute("aria-label", isDark ? "Ganti ke Mode Terang" : "Ganti ke Mode Gelap");
    toggleBtn.setAttribute("title", isDark ? "Ganti ke Mode Terang" : "Ganti ke Mode Gelap");
  }
}

function initTheme() {
  const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === "light" || storedTheme === "dark") {
    applyTheme(storedTheme);
  } else {
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme = prefersDark ? "dark" : "light";
    applyTheme(initialTheme);
  }

  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
      if (!localStorage.getItem(THEME_STORAGE_KEY)) {
        applyTheme(e.matches ? "dark" : "light");
      }
    });
  }
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_STORAGE_KEY, newTheme);
  applyTheme(newTheme);
}

function updateClock() {
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0] + " WIB";
  const clockEl = document.getElementById("systemClock");
  if (clockEl) {
    clockEl.textContent = timeStr;
  }
}

// Immediate execution & interval setup
if (typeof window !== "undefined") {
  updateClock();
  document.addEventListener("DOMContentLoaded", updateClock);
  setInterval(updateClock, 1000);
}

function toggleFullscreen() {
  const videoCard = document.getElementById("videoContainer");
  if (!videoCard) return;

  if (!document.fullscreenElement) {
    videoCard.requestFullscreen().catch((err) => {
      console.warn("Fullscreen request error:", err);
    });
  } else {
    document.exitFullscreen().catch((err) => {
      console.warn("Exit fullscreen error:", err);
    });
  }
}
