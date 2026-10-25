/**
 * FloodDetec AI - Client-Side Application Logic
 * Modern Studio Workbench (Vanilla JavaScript ES6, Zero Dependencies)
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const body = document.body;
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const samplesGrid = document.getElementById("samplesGrid");
  const thresholdInput = document.getElementById("thresholdInput");
  const thresholdValue = document.getElementById("thresholdValue");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const errorMessageText = document.getElementById("errorMessageText");
  const retryBtn = document.getElementById("retryBtn");

  const comparisonContainer = document.getElementById("comparisonContainer");
  const visualizerStage = document.getElementById("visualizerStage");
  const imgOriginal = document.getElementById("imgOriginal");
  const imgOverlay = document.getElementById("imgOverlay");
  const imgBinary = document.getElementById("imgBinary");
  const splitLayerWrapper = document.getElementById("splitLayerWrapper");
  const imgSplitOverlay = document.getElementById("imgSplitOverlay");
  const splitDivider = document.getElementById("splitDivider");
  const modeTabs = document.querySelectorAll(".mode-tab");
  const downloadBtn = document.getElementById("downloadBtn");

  const riskBadge = document.getElementById("riskBadge");
  const riskBadgeDot = document.getElementById("riskBadgeDot");
  const riskBadgeText = document.getElementById("riskBadgeText");
  const floodPercentVal = document.getElementById("floodPercentVal");
  const floodPixelsVal = document.getElementById("floodPixelsVal");
  const latencyVal = document.getElementById("latencyVal");

  // --- State Variables ---
  let activeViewMode = "split"; // 'split', 'overlay', 'binary', 'original'
  let splitPosition = 50; // percentage
  let isDraggingSplit = false;
  let currentResultData = null;
  let recalculateDebounceTimer = null;
  let lastUploadedFile = null;

  // --- 1. Theme Toggle Management ---
  const savedTheme = localStorage.getItem("geoflood_theme") || "dark";
  body.setAttribute("data-theme", savedTheme);

  themeToggleBtn.addEventListener("click", () => {
    const currentTheme = body.getAttribute("data-theme");
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    body.setAttribute("data-theme", nextTheme);
    localStorage.setItem("geoflood_theme", nextTheme);
  });

  // --- 2. Load Sample Gallery (Horizontal Quick-Bar) ---
  async function loadSamples() {
    try {
      const response = await fetch("/api/samples");
      if (!response.ok) throw new Error("Gagal memuat sampel citra");
      const data = await response.json();
      renderSamples(data.samples || []);
    } catch (err) {
      console.warn("Could not load samples:", err);
      samplesGrid.innerHTML = '<span class="upload-hint">Sampel tidak tersedia</span>';
    }
  }

  function renderSamples(samples) {
    if (!samples.length) {
      samplesGrid.innerHTML = '<span class="upload-hint">Belum ada sampel terdaftar</span>';
      return;
    }

    samplesGrid.innerHTML = "";
    samples.forEach((sample, idx) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "sample-pill-card";
      card.setAttribute("aria-label", `Uji sampel ${idx + 1}: ${sample.title}`);

      card.innerHTML = `
        <img src="${sample.url}" alt="${sample.title}" class="sample-pill-thumb" loading="lazy">
        <div class="sample-pill-content">
          <span class="sample-pill-title">${sample.title}</span>
          <span class="sample-pill-sub">${sample.expected_risk || "Uji AI"}</span>
        </div>
      `;

      card.addEventListener("click", () => {
        document.querySelectorAll(".sample-pill-card").forEach(c => c.classList.remove("active"));
        card.classList.add("active");
        loadSampleImageFromUrl(sample.url, sample.filename || `sample_${idx + 1}.jpg`);
      });

      samplesGrid.appendChild(card);
    });
  }

  async function loadSampleImageFromUrl(url, filename) {
    showLoadingState("Mengunduh citra sampel...");
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("Gagal mengambil file sampel");
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      executeSegmentation(file);
    } catch (err) {
      showErrorState(err.message || "Gagal memuat citra sampel.");
    }
  }

  // --- 3. Drag and Drop File Handlers ---
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach(evtName => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(evtName => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      // Remove active state from sample pills when custom image uploaded
      document.querySelectorAll(".sample-pill-card").forEach(c => c.classList.remove("active"));
      handleSelectedFile(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      document.querySelectorAll(".sample-pill-card").forEach(c => c.classList.remove("active"));
      handleSelectedFile(fileInput.files[0]);
    }
  });

  function handleSelectedFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
      showErrorState("Format file tidak didukung. Harap pilih citra JPG atau PNG.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showErrorState("Ukuran file melebihi batas maksimal 10 MB.");
      return;
    }
    executeSegmentation(file);
  }

  // --- 4. Segmentation Request Execution ---
  async function executeSegmentation(file) {
    lastUploadedFile = file;
    showLoadingState("Mengeksekusi model Attention U-Net...");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("threshold", thresholdInput.value);

    try {
      const response = await fetch("/api/segment", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Terjadi kesalahan saat inferensi citra.");
      }

      const res = await response.json();
      currentResultData = res.data;
      displayResults(res.data);
    } catch (err) {
      showErrorState(err.message || "Gagal menghubungi server inferensi.");
    }
  }

  // --- 5. Real-Time Threshold Adjustment ---
  thresholdInput.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    thresholdValue.textContent = val;
    thresholdInput.setAttribute("aria-valuenow", val);

    if (!currentResultData) return;

    clearTimeout(recalculateDebounceTimer);
    recalculateDebounceTimer = setTimeout(async () => {
      try {
        const response = await fetch("/api/recalculate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threshold: parseFloat(val) })
        });
        if (response.ok) {
          const res = await response.json();
          currentResultData = res.data;
          updateMetricsAndMasksOnly(res.data);
        }
      } catch (err) {
        console.warn("Recalculate failed:", err);
      }
    }, 100);
  });

  // --- 6. Results Display and Mode Switching ---
  function displayResults(data) {
    hideAllStates();
    comparisonContainer.classList.remove("hidden");
    downloadBtn.disabled = false;

    // Set Image Sources
    imgOriginal.src = data.images.original_url;
    imgOverlay.src = data.images.overlay_url;
    imgBinary.src = data.images.binary_url || data.images.binary_mask_url;
    imgSplitOverlay.src = data.images.overlay_url;

    // Sync split image sizes once image is loaded
    imgOriginal.onload = () => syncSplitImageSize();

    updateMetricsAndMasksOnly(data);
    applyViewMode(activeViewMode);
    syncSplitImageSize();
  }

  function syncSplitImageSize() {
    const w = comparisonContainer.clientWidth;
    const h = comparisonContainer.clientHeight;
    if (w > 0 && h > 0) {
      imgSplitOverlay.style.width = `${w}px`;
      imgSplitOverlay.style.height = `${h}px`;
    }
  }

  window.addEventListener("resize", () => {
    if (currentResultData) syncSplitImageSize();
  });

  function updateMetricsAndMasksOnly(data) {
    // Update Image Sources for dynamic masks
    imgOverlay.src = data.images.overlay_url;
    imgBinary.src = data.images.binary_url || data.images.binary_mask_url;
    imgSplitOverlay.src = data.images.overlay_url;

    // Update Numerical Readouts
    floodPercentVal.textContent = `${data.flood_percentage.toFixed(2)}%`;
    floodPixelsVal.textContent = `${data.flood_pixel_count.toLocaleString()} / ${data.total_pixel_count.toLocaleString()} px`;
    if (data.inference_time_ms !== undefined) {
      latencyVal.textContent = `${data.inference_time_ms} ms`;
    }

    // Update Severity Risk Badge
    riskBadge.setAttribute("data-level", data.risk_level);
    riskBadgeText.textContent = data.risk_level;
  }

  // --- 7. View Tabs Management ---
  modeTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      modeTabs.forEach(t => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");

      activeViewMode = tab.getAttribute("data-mode");
      applyViewMode(activeViewMode);
    });
  });

  function applyViewMode(mode) {
    if (!currentResultData) return;

    // Hide all view layers first
    imgOverlay.classList.add("hidden");
    imgBinary.classList.add("hidden");
    splitLayerWrapper.style.display = "none";
    splitDivider.style.display = "none";
    imgOriginal.style.display = "none";

    const splitGuides = document.getElementById("splitGuides");
    if (splitGuides) splitGuides.style.display = "none";

    if (mode === "split") {
      imgOriginal.style.display = "block";
      splitLayerWrapper.style.display = "block";
      splitDivider.style.display = "flex";
      if (splitGuides) splitGuides.style.display = "flex";
      setSplitPosition(splitPosition);
      syncSplitImageSize();
    } else if (mode === "overlay") {
      imgOverlay.classList.remove("hidden");
    } else if (mode === "binary") {
      imgBinary.classList.remove("hidden");
    } else if (mode === "original") {
      imgOriginal.style.display = "block";
    }
  }

  // --- 8. Interactive Split Slider Handlers ---
  function setSplitPosition(pos) {
    splitPosition = Math.max(0, Math.min(100, pos));
    splitLayerWrapper.style.width = `${splitPosition}%`;
    splitDivider.style.left = `${splitPosition}%`;
    splitDivider.setAttribute("aria-valuenow", Math.round(splitPosition));
  }

  function handleSliderMove(clientX) {
    const rect = comparisonContainer.getBoundingClientRect();
    if (rect.width <= 0) return;
    const x = clientX - rect.left;
    const pos = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSplitPosition(pos);
  }

  splitDivider.addEventListener("mousedown", (e) => {
    e.preventDefault();
    isDraggingSplit = true;
  });

  window.addEventListener("mousemove", (e) => {
    if (isDraggingSplit) {
      e.preventDefault();
      handleSliderMove(e.clientX);
    }
  });

  window.addEventListener("mouseup", () => {
    isDraggingSplit = false;
  });

  // Touch Support for mobile
  splitDivider.addEventListener("touchstart", () => {
    isDraggingSplit = true;
  }, { passive: true });

  window.addEventListener("touchmove", (e) => {
    if (isDraggingSplit && e.touches.length > 0) {
      handleSliderMove(e.touches[0].clientX);
    }
  }, { passive: true });

  window.addEventListener("touchend", () => {
    isDraggingSplit = false;
  });

  // Keyboard navigation on Split Divider
  splitDivider.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      setSplitPosition(splitPosition - 5);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      setSplitPosition(splitPosition + 5);
    } else if (e.key === "Home") {
      e.preventDefault();
      setSplitPosition(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setSplitPosition(100);
    }
  });

  // --- 9. Download Functionality ---
  downloadBtn.addEventListener("click", () => {
    if (!currentResultData) return;

    let targetUrl = currentResultData.images.overlay_url;
    let filenameSuffix = "overlay";

    if (activeViewMode === "binary") {
      targetUrl = currentResultData.images.binary_url || currentResultData.images.binary_mask_url;
      filenameSuffix = "binary_mask";
    } else if (activeViewMode === "original") {
      targetUrl = currentResultData.images.original_url;
      filenameSuffix = "original";
    }

    const a = document.createElement("a");
    a.href = targetUrl;
    a.download = `flooddetec_${filenameSuffix}_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  // --- 10. UI State Switchers ---
  function hideAllStates() {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    errorState.classList.add("hidden");
  }

  function showLoadingState(message = "Memproses inferensi...") {
    hideAllStates();
    comparisonContainer.classList.add("hidden");
    downloadBtn.disabled = true;
    document.getElementById("loadingStatusText").textContent = message;
    loadingState.classList.remove("hidden");
  }

  function showErrorState(message) {
    hideAllStates();
    comparisonContainer.classList.add("hidden");
    downloadBtn.disabled = true;
    errorMessageText.textContent = message;
    errorState.classList.remove("hidden");
  }

  retryBtn.addEventListener("click", () => {
    if (lastUploadedFile) {
      executeSegmentation(lastUploadedFile);
    } else {
      hideAllStates();
      emptyState.classList.remove("hidden");
    }
  });

  // Initialize
  setSplitPosition(50);
  loadSamples();
});
