/* ═══════════════════════════════════════════════════════════════════════════
   VIGIL — Mission Control Center Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── Global State ──────────────────────────────────────────────────────────
  let experiments = [];
  let activePhaseId = null;
  let activeExperiment = null;
  let missionRunning = false;
  let stepStatuses = {};       // { step_id: 'pending' | 'in_progress' | 'done' | 'skipped' }
  let stepConfidences = {};    // { step_id: 0.0 - 1.0 }
  let currentSources = null;

  // Browser Webcam State
  let currentSourceMode = "browser"; // "browser" or "server"
  let localMediaStream = null;
  let browserCaptureInterval = null;

  // ── DOM Element Cache ─────────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const catalogList = $("#catalog-list");
  const catalogCountBadge = $("#catalog-count-badge");
  const stepList = $("#step-list");
  const experimentTitle = $("#experiment-title");
  const experimentMeta = $("#experiment-meta");
  const quickOverrides = $("#quick-overrides");
  const timelineBody = $("#timeline-body");
  const timelineEmpty = $("#timeline-empty");
  const btnStart = $("#btn-start-mission");
  const btnReset = $("#btn-reset-mission");
  const btnPassStep = $("#btn-pass-step");
  const btnSkipStep = $("#btn-skip-step");
  const btnDownloadJson = $("#btn-download-json");
  const btnDownloadCsv = $("#btn-download-csv");
  const recBadge = $("#rec-badge");
  const systemPill = $("#system-pill");
  const systemLabel = $("#system-label");
  const voiceToggleInput = $("#voice-toggle-input");
  const resetConfirm = $("#reset-confirm");
  const toastContainer = $("#toast-container");
  const videoSourceSelect = $("#video-source-select");
  const btnToggleOverlay = $("#btn-toggle-overlay");
  const btnSnapshot = $("#btn-snapshot");
  const topnavSourceLabel = $("#topnav-source-label");

  // Camera elements
  const browserWebcamVideo = $("#browser-webcam-video");
  const browserCaptureCanvas = $("#browser-capture-canvas");
  const videoFeedImg = $("#video-feed");
  const cameraPermBanner = $("#camera-perm-banner");
  const btnRequestCamPerm = $("#btn-request-cam-perm");

  // Telemetry HUD Elements
  const hudActionBadge = $("#hud-action-badge");
  const telemGrasp = $("#telem-grasp");
  const telemOverlap = $("#telem-overlap");
  const telemHands = $("#telem-hands");
  const telemSurface = $("#telem-surface");
  const telemTimerVal = $("#telem-timer-val");
  const telemTimerFill = $("#telem-timer-fill");
  const radarChips = $("#radar-chips");
  const streamFpsVal = $("#fps-val");

  // ── Socket.IO Connection ──────────────────────────────────────────────────
  const socket = io();

  socket.on("connect", () => {
    console.log("[VIGIL] Socket.IO Telemetry Link Connected");
    syncMissionState();
  });

  function syncMissionState() {
    fetch("/api/mission/state")
      .then((r) => r.json())
      .then((state) => {
        // STRICT CHECK: Mission is ONLY running if backend explicitly reports is_running === true
        if (state.is_running === true) {
          missionRunning = true;
          if (state.phase_id) {
            activePhaseId = state.phase_id;
            renderCatalog();
            loadActiveExperiment();
          }
          if (state.step_statuses) {
            stepStatuses = state.step_statuses;
            renderStepList();
          }
        } else {
          missionRunning = false;
        }
        updateMissionUI();
      })
      .catch((err) => console.warn("[VIGIL] Failed syncing mission state:", err));
  }

  // ── Real-Time Telemetry Stream Handler ────────────────────────────────────
  socket.on("telemetry_update", (telem) => {
    // 1. Action Badge
    if (hudActionBadge) {
      hudActionBadge.textContent = telem.action || "IDLE";
      hudActionBadge.className = "hud-status-badge " + (telem.action ? telem.action.toLowerCase() : "idle");
    }

    // 2. Grasp & Overlap Meters
    if (telemGrasp) {
      telemGrasp.textContent = telem.grasp_active ? "GRASPED" : "OPEN";
      telemGrasp.className = "telem-value " + (telem.grasp_active ? "good" : "");
    }
    if (telemOverlap) {
      telemOverlap.textContent = `${Number(telem.bbox_overlap || 0).toFixed(1)}%`;
      telemOverlap.className = "telem-value " + (telem.bbox_overlap > 15 ? "good" : "");
    }
    if (telemHands) {
      telemHands.textContent = telem.hands_count || "0";
    }
    if (telemSurface) {
      telemSurface.textContent = telem.target_surface ? telem.target_surface.replace(/_/g, " ") : "—";
    }

    // 3. Inspection Hold Timer
    if (telemTimerVal && telemTimerFill) {
      if (telem.timer_active && telem.timer_elapsed > 0) {
        telemTimerVal.textContent = `${telem.timer_elapsed.toFixed(1)} / ${telem.timer_duration.toFixed(1)} s`;
        telemTimerFill.style.width = `${Math.min(100, telem.timer_progress)}%`;
        const timerBox = $("#inspection-timer-box");
        if (timerBox) timerBox.classList.add("active");
      } else {
        telemTimerVal.textContent = `0.0 / ${telem.timer_duration ? telem.timer_duration.toFixed(1) : '5.0'} s`;
        telemTimerFill.style.width = "0%";
        const timerBox = $("#inspection-timer-box");
        if (timerBox) timerBox.classList.remove("active");
      }
    }

    // 4. Measured FPS
    if (streamFpsVal && telem.fps !== undefined) {
      streamFpsVal.textContent = Math.round(telem.fps);
    }

    // 5. Detected Objects Radar Chips
    if (radarChips && telem.detected_objects) {
      if (telem.detected_objects.length === 0) {
        radarChips.innerHTML = '<span class="chip chip-dim">No payload objects in view</span>';
      } else {
        radarChips.innerHTML = telem.detected_objects
          .slice(0, 8)
          .map((obj) => {
            const cleanName = obj.name.replace(/_/g, " ");
            return `<span class="chip chip-${obj.name.toLowerCase()}">${cleanName} <strong>${obj.confidence.toFixed(0)}%</strong></span>`;
          })
          .join("");
      }
    }
  });

  // ── Step Update Handler ───────────────────────────────────────────────────
  socket.on("step_update", (data) => {
    // Only update steps if a mission is running or was running
    if (missionRunning) {
      stepStatuses[data.step_id] = data.status;
      if (data.confidence !== undefined) {
        stepConfidences[data.step_id] = data.confidence;
      }
      renderStepList();
    }
  });

  // ── Timeline Row Handler ──────────────────────────────────────────────────
  socket.on("timeline_row", (row) => {
    addTimelineRow(row);
  });

  // ── Alerts & Voice Synthesizer ────────────────────────────────────────────
  socket.on("alert", (data) => {
    showToast(data.message, data.type || "warning");
    if (voiceToggleInput && voiceToggleInput.checked) {
      speak(data.message);
    }
  });

  // ── System Status Handler ─────────────────────────────────────────────────
  socket.on("system_status", (data) => {
    updateSystemPill(data.status, data.label);
    if (data.status === "running") {
      missionRunning = true;
    } else {
      missionRunning = false;
    }
    updateMissionUI();
  });

  // ── Browser Camera Permission & Streaming ─────────────────────────────────

  async function startBrowserWebcam() {
    currentSourceMode = "browser";
    if (cameraPermBanner) cameraPermBanner.style.display = "none";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast("Browser camera API (getUserMedia) not supported in this browser. Switching to Server Stream.", "danger");
      switchToServerSource("camera", 1);
      return;
    }

    try {
      showToast("Requesting camera permission…", "warning");

      // Triggers the native browser camera permission prompt!
      localMediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user"
        },
        audio: false
      });

      if (browserWebcamVideo) {
        browserWebcamVideo.srcObject = localMediaStream;
        browserWebcamVideo.style.display = "block";
      }
      if (videoFeedImg) {
        videoFeedImg.style.display = "none";
      }

      showToast("Camera permission granted! Live optical feed active.", "success");
      if (topnavSourceLabel) topnavSourceLabel.textContent = "Browser Webcam";

      // Begin streaming captured frames from video element to AI detector
      startBrowserFrameStreaming();
    } catch (err) {
      console.warn("Camera access error:", err);
      if (cameraPermBanner) cameraPermBanner.style.display = "flex";
      showToast(`Camera permission required: ${err.message || err.name}. Click Allow to grant access.`, "danger");
    }
  }

  function startBrowserFrameStreaming() {
    if (browserCaptureInterval) clearInterval(browserCaptureInterval);
    if (!browserWebcamVideo || !browserCaptureCanvas) return;

    const ctx = browserCaptureCanvas.getContext("2d");
    let isSending = false;

    // Stream frames at ~18 FPS to the AI backend
    browserCaptureInterval = setInterval(() => {
      if (!localMediaStream || browserWebcamVideo.paused || browserWebcamVideo.ended || browserWebcamVideo.videoWidth === 0) return;
      if (isSending) return; // Prevent network backlog

      browserCaptureCanvas.width = browserWebcamVideo.videoWidth || 640;
      browserCaptureCanvas.height = browserWebcamVideo.videoHeight || 480;
      ctx.drawImage(browserWebcamVideo, 0, 0, browserCaptureCanvas.width, browserCaptureCanvas.height);

      browserCaptureCanvas.toBlob((blob) => {
        if (!blob) return;
        isSending = true;
        const formData = new FormData();
        formData.append("frame", blob, "browser_webcam.jpg");

        fetch("/api/process_frame", {
          method: "POST",
          body: formData,
        })
          .then((r) => r.json())
          .then(() => {
            isSending = false;
          })
          .catch(() => {
            isSending = false;
          });
      }, "image/jpeg", 0.72);
    }, 55);
  }

  function switchToServerSource(sourceType, sourceVal) {
    currentSourceMode = "server";

    // Stop browser camera stream
    if (browserCaptureInterval) {
      clearInterval(browserCaptureInterval);
      browserCaptureInterval = null;
    }
    if (localMediaStream) {
      localMediaStream.getTracks().forEach((track) => track.stop());
      localMediaStream = null;
    }

    if (cameraPermBanner) cameraPermBanner.style.display = "none";
    if (browserWebcamVideo) browserWebcamVideo.style.display = "none";
    if (videoFeedImg) {
      videoFeedImg.src = "/video_feed?x=" + Date.now();
      videoFeedImg.style.display = "block";
    }

    fetch("/api/camera/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_type: sourceType, source_val: sourceVal }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          showToast(data.message, "success");
          loadAvailableCameras();
        } else {
          showToast(data.error || "Failed switching source", "danger");
        }
      })
      .catch(() => showToast("Network error switching source", "danger"));
  }

  // Camera permission retry button
  if (btnRequestCamPerm) {
    btnRequestCamPerm.addEventListener("click", () => {
      startBrowserWebcam();
    });
  }

  // Source selector dropdown change
  if (videoSourceSelect) {
    videoSourceSelect.addEventListener("change", () => {
      const val = videoSourceSelect.value;
      if (val === "browser:webcam") {
        startBrowserWebcam();
      } else {
        const [type, sourceVal] = val.split(":");
        switchToServerSource(type, sourceVal);
      }
    });
  }

  // ── Load Experiments Catalog ──────────────────────────────────────────────
  function loadExperiments() {
    fetch("/api/experiments")
      .then((r) => r.json())
      .then((data) => {
        experiments = data;
        if (catalogCountBadge) catalogCountBadge.textContent = experiments.length;
        renderCatalog();
        renderReorderList();
        renderHardwareStatus();

        if (!activePhaseId && experiments.length > 0) {
          selectPhase(experiments[0].id);
        }
      })
      .catch((err) => console.error("Error loading experiments:", err));
  }

  // ── Catalog Rendering ─────────────────────────────────────────────────────
  function renderCatalog() {
    if (!catalogList) return;
    const phaseIcons = [
      '<i class="ph ph-flask"></i>',
      '<i class="ph ph-mouse-simple"></i>',
      '<i class="ph ph-headphones"></i>',
      '<i class="ph ph-squares-four"></i>',
      '<i class="ph ph-tree-structure"></i>',
      '<i class="ph ph-eye"></i>',
      '<i class="ph ph-check-circle"></i>',
    ];

    catalogList.innerHTML = experiments
      .map((exp, i) => {
        const isActive = exp.id === activePhaseId;
        const icon = phaseIcons[i % phaseIcons.length];
        const stepCount = exp.steps ? exp.steps.length : 0;
        return `
          <div class="catalog-item ${isActive ? "active" : ""}"
               data-phase-id="${exp.id}" onclick="VIGIL.selectPhase('${exp.id}')">
            <div class="catalog-item-icon">${icon}</div>
            <div class="catalog-item-name">${escHtml(exp.name)}</div>
            <div class="catalog-item-tag">${stepCount} steps</div>
          </div>`;
      })
      .join("");
  }

  function selectPhase(phaseId) {
    activePhaseId = phaseId;
    renderCatalog();
    loadActiveExperiment();
    btnStart.disabled = missionRunning;

    // Highlight in Set Mission view
    $$(".reorder-item").forEach((el) => {
      el.style.borderColor = el.dataset.phaseId === phaseId ? "rgba(140,124,251,0.6)" : "var(--panel-border)";
    });
    const btnSetActive = $("#btn-set-active");
    if (btnSetActive) btnSetActive.disabled = false;
  }

  function loadActiveExperiment() {
    activeExperiment = experiments.find((e) => e.id === activePhaseId) || null;
    if (activeExperiment) {
      if (experimentTitle) experimentTitle.textContent = activeExperiment.name;
      if (experimentMeta) {
        experimentMeta.textContent = `${activeExperiment.steps.length} sequential protocol steps · Ready`;
      }
      activeExperiment.steps.forEach((s) => {
        if (!missionRunning) {
          stepStatuses[s.id] = "pending";
          stepConfidences[s.id] = 0.0;
        } else {
          if (!(s.id in stepStatuses)) stepStatuses[s.id] = "pending";
          if (!(s.id in stepConfidences)) stepConfidences[s.id] = 0.0;
        }
      });
      renderStepList();
      renderStepInspector(activeExperiment);
    }
  }

  // ── Step List Rendering ───────────────────────────────────────────────────
  function renderStepList() {
    if (!activeExperiment || !stepList) return;
    stepList.innerHTML = activeExperiment.steps
      .map((step) => {
        const status = stepStatuses[step.id] || "pending";
        const conf = stepConfidences[step.id] || 0.0;
        const pct = status === "done" || status === "skipped" ? 100 : Math.round(conf * 100);
        const badgeText = {
          pending: "Pending",
          in_progress: "In Progress",
          done: "Verified",
          skipped: "Skipped",
        }[status] || status;

        const actionTag = step.expected_action ? `<span class="step-action-tag">${escHtml(step.expected_action)}</span>` : "";
        const targetTag = step.target_surface ? `<span class="step-target-tag">→ ${escHtml(step.target_surface.replace(/_/g, " "))}</span>` : "";

        return `
          <div class="step-row ${status}">
            <div class="step-num-col">
              <span class="step-num">Step ${step.id}</span>
              ${actionTag}
            </div>
            <div class="step-center-col">
              <div class="step-label">${escHtml(step.label)} ${targetTag}</div>
              <div class="step-progress-bar">
                <div class="step-progress-fill" style="width:${pct}%"></div>
              </div>
            </div>
            <div class="step-end-col">
              <span class="step-badge ${status}">${badgeText}</span>
            </div>
          </div>`;
      })
      .join("");
  }

  // ── Protocol Step Inspector (for Set Mission View) ────────────────────────
  function renderStepInspector(exp) {
    const container = $("#protocol-step-inspector");
    if (!container || !exp) return;

    container.innerHTML = `
      <div style="margin-bottom:12px;">
        <h3 style="font-family:var(--font-serif);font-size:18px;color:var(--text-primary);">${escHtml(exp.name)}</h3>
        <p style="font-size:12px;color:var(--text-muted);margin-top:2px;">Protocol ID: <code>${exp.id}</code> · ${exp.steps.length} sequential execution stages</p>
      </div>
      <div class="inspector-step-table">
        ${exp.steps.map((st) => `
          <div class="inspector-step-item">
            <div class="inspector-step-num">${st.id}</div>
            <div class="inspector-step-details">
              <div class="inspector-step-title">${escHtml(st.label)}</div>
              <div class="inspector-step-tags">
                <span class="tag">Object: <strong>${escHtml(st.expected_object || 'any')}</strong></span>
                <span class="tag">Action: <strong>${escHtml(st.expected_action || 'VERIFY')}</strong></span>
                ${st.duration ? `<span class="tag">Duration: <strong>${st.duration}s</strong></span>` : ''}
                ${st.target_surface ? `<span class="tag">Target: <strong>${escHtml(st.target_surface)}</strong></span>` : ''}
              </div>
            </div>
          </div>`).join("")}
      </div>`;
  }

  // ── Timeline Row Insertion ────────────────────────────────────────────────
  function addTimelineRow(row) {
    if (timelineEmpty) timelineEmpty.style.display = "none";
    if (!timelineBody) return;

    const tr = document.createElement("tr");
    const isSuccess = !!row.success;
    const successClass = isSuccess ? "yes" : "no";
    const successText = isSuccess ? '<i class="ph ph-check-circle"></i> VERIFIED' : '<i class="ph ph-x-circle"></i> SKIPPED';
    const accPct = `${(Number(row.accuracy || 0) * 100).toFixed(1)}%`;

    tr.innerHTML = `
      <td>${escHtml(truncate(row.phase || "", 20))}</td>
      <td><strong>${row.step}</strong></td>
      <td>${escHtml(truncate(row.step_label || "", 30))}</td>
      <td class="col-success ${successClass}">${successText}</td>
      <td><strong>${accPct}</strong></td>
      <td>${formatTime(row.end_time || row.start_time)}</td>`;

    timelineBody.appendChild(tr);

    // Auto-scroll timeline container to latest row
    const container = timelineBody.closest(".panel-body");
    if (container) container.scrollTop = container.scrollHeight;
  }

  // ── Mission Controls (Explicit user trigger ONLY) ─────────────────────────
  btnStart.addEventListener("click", () => {
    if (!activePhaseId || missionRunning) return;

    // Reset step statuses for a fresh run
    stepStatuses = {};
    stepConfidences = {};
    if (timelineBody) timelineBody.innerHTML = "";
    if (timelineEmpty) timelineEmpty.style.display = "flex";

    fetch("/api/mission/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phase_id: activePhaseId }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          missionRunning = true;
          updateMissionUI();
          loadActiveExperiment();
          showToast(`Mission started: ${activeExperiment.name}`, "success");
        } else {
          showToast(data.error || "Failed to start mission", "danger");
        }
      })
      .catch(() => showToast("Server communication error", "danger"));
  });

  btnReset.addEventListener("click", () => {
    if (missionRunning) {
      resetConfirm.classList.add("visible");
    } else {
      doReset();
    }
  });

  $("#btn-reset-confirm-yes").addEventListener("click", () => {
    resetConfirm.classList.remove("visible");
    doReset();
  });

  $("#btn-reset-confirm-no").addEventListener("click", () => {
    resetConfirm.classList.remove("visible");
  });

  function doReset() {
    fetch("/api/mission/reset", { method: "POST" })
      .then((r) => r.json())
      .then(() => {
        missionRunning = false;
        stepStatuses = {};
        stepConfidences = {};
        renderStepList();
        updateMissionUI();
        showToast("Mission reset — system in standby", "success");
      });
  }

  // Quick Overrides (Pass / Skip Step)
  if (btnPassStep) {
    btnPassStep.addEventListener("click", () => {
      fetch("/api/mission/pass_step", { method: "POST" })
        .then((r) => r.json())
        .then((data) => {
          if (data.ok) showToast(data.message, "success");
          else showToast(data.error || "Cannot pass step", "warning");
        });
    });
  }

  if (btnSkipStep) {
    btnSkipStep.addEventListener("click", () => {
      fetch("/api/mission/skip_step", { method: "POST" })
        .then((r) => r.json())
        .then((data) => {
          if (data.ok) showToast(data.message, "warning");
          else showToast(data.error || "Cannot skip step", "warning");
        });
    });
  }

  $("#btn-clear-timeline").addEventListener("click", () => {
    if (timelineBody) timelineBody.innerHTML = "";
    if (timelineEmpty) timelineEmpty.style.display = "flex";
    showToast("Local timeline cleared", "success");
  });

  function updateMissionUI() {
    btnStart.disabled = missionRunning || !activePhaseId;
    btnReset.disabled = false;

    if (missionRunning) {
      btnStart.innerHTML = '<span class="spinner"></span> Mission Active…';
      if (recBadge) recBadge.classList.add("visible");
      if (quickOverrides) quickOverrides.style.display = "flex";
    } else {
      btnStart.innerHTML = '<span class="btn-icon"><i class="ph ph-play"></i></span> Start Mission';
      if (recBadge) recBadge.classList.remove("visible");
      if (quickOverrides) quickOverrides.style.display = "none";
    }
  }

  // ── Optical Sources API Sync ─────────────────────────────────────────────
  function loadAvailableCameras() {
    fetch("/api/cameras")
      .then((r) => r.json())
      .then((data) => {
        currentSources = data;
        if (!videoSourceSelect) return;

        let optionsHtml = `
          <optgroup label="Client Optical Stream">
            <option value="browser:webcam" ${currentSourceMode === "browser" ? "selected" : ""}>Browser Webcam (Requests Permission)</option>
          </optgroup>`;

        // Physical Cameras
        if (data.cameras && data.cameras.length > 0) {
          optionsHtml += `<optgroup label="Server Physical Cameras">`;
          data.cameras.forEach((cam) => {
            const isSel = currentSourceMode === "server" && data.current.type === "camera" && String(data.current.value) === String(cam.id);
            optionsHtml += `<option value="camera:${cam.id}" ${isSel ? "selected" : ""}>Server · ${escHtml(cam.name)} (${cam.backend})</option>`;
          });
          optionsHtml += `</optgroup>`;
        }

        // Test Videos
        if (data.videos && data.videos.length > 0) {
          optionsHtml += `<optgroup label="Workspace Video Playback">`;
          data.videos.forEach((vid) => {
            const isSel = currentSourceMode === "server" && data.current.type === "video" && String(data.current.value) === String(vid.name);
            optionsHtml += `<option value="video:${vid.name}" ${isSel ? "selected" : ""}>Video: ${escHtml(vid.name)} (${vid.size_mb} MB)</option>`;
          });
          optionsHtml += `</optgroup>`;
        }

        optionsHtml += `<optgroup label="Synthetic">
          <option value="placeholder:synthetic">Synthetic Radar Screen</option>
        </optgroup>`;

        videoSourceSelect.innerHTML = optionsHtml;
      })
      .catch((err) => console.warn("[VIGIL] Error fetching cameras:", err));
  }

  // Toggle Overlay
  if (btnToggleOverlay) {
    btnToggleOverlay.addEventListener("click", () => {
      fetch("/api/camera/toggle_overlay", { method: "POST" })
        .then((r) => r.json())
        .then((data) => {
          btnToggleOverlay.classList.toggle("active", data.overlay_active);
          showToast(`AI HUD Overlay: ${data.overlay_active ? "ENABLED" : "DISABLED"}`, "success");
        });
    });
  }

  // Snapshot Capture
  if (btnSnapshot) {
    btnSnapshot.addEventListener("click", () => {
      const link = document.createElement("a");
      if (currentSourceMode === "browser" && browserCaptureCanvas) {
        link.href = browserCaptureCanvas.toDataURL("image/jpeg", 0.95);
      } else {
        link.href = "/snapshot?x=" + Date.now();
      }
      link.download = `vigil-snapshot-${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      showToast("Still frame snapshot captured and downloaded", "success");
    });
  }

  // ── Download Logs (JSON & CSV) ────────────────────────────────────────────
  if (btnDownloadJson) {
    btnDownloadJson.addEventListener("click", () => {
      window.location.href = "/api/logs/download";
    });
  }

  if (btnDownloadCsv) {
    btnDownloadCsv.addEventListener("click", () => {
      window.location.href = "/api/logs/download/csv";
    });
  }

  // ── Voice Synthesizer ─────────────────────────────────────────────────────
  const savedVoice = localStorage.getItem("vigil-voice-alerts");
  if (voiceToggleInput) {
    voiceToggleInput.checked = savedVoice === null ? true : savedVoice === "true";
    voiceToggleInput.addEventListener("change", () => {
      localStorage.setItem("vigil-voice-alerts", voiceToggleInput.checked);
    });
  }

  function speak(text) {
    if (!("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.05;
      utter.pitch = 1.0;
      utter.volume = 0.85;
      window.speechSynthesis.speak(utter);
    } catch (e) {
      console.warn("Speech error:", e);
    }
  }

  // ── Top Navigation View Switching ─────────────────────────────────────────
  $$(".nav-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const viewName = pill.dataset.view;
      $$(".nav-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      $$(".view").forEach((v) => v.classList.remove("active"));
      const target = document.getElementById("view-" + viewName);
      if (target) target.classList.add("active");

      if (viewName === "hardware") refreshHardware();
      if (viewName === "calibration") refreshCalibration();
      if (viewName === "set-mission") renderReorderList();
    });
  });

  // ── Set Mission View ──────────────────────────────────────────────────────
  function renderReorderList() {
    const list = $("#reorder-list");
    if (!list) return;
    list.innerHTML = experiments
      .map((exp) => {
        const isSel = exp.id === activePhaseId;
        return `
          <div class="reorder-item" data-phase-id="${exp.id}"
               style="border-color:${isSel ? "rgba(140,124,251,0.6)" : "var(--panel-border)"}; cursor:pointer;"
               onclick="VIGIL.selectPhase('${exp.id}')">
            <span class="reorder-handle"><i class="ph ph-dots-six-vertical"></i></span>
            <span class="reorder-name">${escHtml(exp.name)}</span>
            <span class="reorder-steps">${exp.steps.length} steps</span>
          </div>`;
      })
      .join("");
  }

  const btnSetActive = $("#btn-set-active");
  if (btnSetActive) {
    btnSetActive.addEventListener("click", () => {
      if (activePhaseId) {
        showToast("Protocol activated: " + (activeExperiment ? activeExperiment.name : activePhaseId), "success");
        document.querySelector('[data-view="dashboard"]').click();
      }
    });
  }

  // ── Calibration View ──────────────────────────────────────────────────────
  function refreshCalibration() {
    fetch("/api/system/status")
      .then((r) => r.json())
      .then((data) => {
        const sourceNameEl = $("#cal-source-name");
        const resEl = $("#cal-resolution");
        const fpsEl = $("#cal-fps");
        const computeEl = $("#cal-compute-device");
        const yoloEl = $("#cal-yolo-status");

        if (sourceNameEl) sourceNameEl.textContent = currentSourceMode === "browser" ? "Browser Webcam (WebRTC)" : (data.camera_name || "Connected");
        if (resEl) resEl.textContent = currentSourceMode === "browser" ? "1280 × 720 (Client)" : (data.resolution || "1280 × 720");
        if (fpsEl) fpsEl.textContent = `${Math.round(data.fps || 24)} fps`;
        if (computeEl) computeEl.textContent = data.gpu_device || "CPU";
        if (yoloEl) yoloEl.textContent = data.detector ? data.model_name : "Degraded / Off";
      });
  }

  const btnRunCalibration = $("#btn-run-calibration");
  if (btnRunCalibration) {
    btnRunCalibration.addEventListener("click", () => {
      btnRunCalibration.innerHTML = '<span class="spinner"></span> Running diagnostics…';
      fetch("/api/system/diagnostics", { method: "POST" })
        .then((r) => r.json())
        .then((res) => {
          btnRunCalibration.innerHTML = '<span class="btn-icon"><i class="ph ph-lightning"></i></span> Run Diagnostics Check';
          showToast(`Diagnostics Completed: Health is ${res.overall_health}`, "success");
          refreshCalibration();
        })
        .catch(() => {
          btnRunCalibration.innerHTML = '<span class="btn-icon"><i class="ph ph-lightning"></i></span> Run Diagnostics Check';
        });
    });
  }

  // ── Hardware Status View ──────────────────────────────────────────────────
  function renderHardwareStatus() {
    refreshHardware();
  }

  function refreshHardware() {
    fetch("/api/system/status")
      .then((r) => r.json())
      .then((data) => {
        const list = $("#hw-list");
        if (!list) return;

        const items = [
          {
            name: "Optical Capture Subsystem",
            detail: currentSourceMode === "browser" ? "Client Browser Webcam (getUserMedia · Active)" : `${data.camera_name} · ${data.resolution} @ ${Math.round(data.fps || 24)} FPS`,
            online: true,
          },
          {
            name: "YOLOv11 Deep Neural Detector",
            detail: `${data.model_name} · Weights: runs/detect/train/weights/best.pt`,
            online: data.detector,
          },
          {
            name: "MediaPipe Hand Landmarking Engine",
            detail: "21 3D Joint Points · Real-time Monotonic Video Mode",
            online: data.mediapipe,
          },
          {
            name: "Neural Compute Acceleration",
            detail: `Device: ${data.gpu_device}`,
            online: true,
          },
          {
            name: "SQLite Persistence Engine",
            detail: "WAL Mode · DB: vigil.db",
            online: true,
          },
          {
            name: "WebSocket Telemetry Synchronizer",
            detail: "Bi-directional Socket.IO · 10 Hz Telemetry Link",
            online: true,
          },
          {
            name: "FSM Action & Hold-Timer Engine",
            detail: "States: IDLE, GRASP, MOVE, RELEASE, PLACED, INSPECT (5s Hold)",
            online: true,
          },
        ];

        list.innerHTML = items
          .map((hw) => `
            <div class="hw-item">
              <div class="hw-dot ${hw.online ? "online" : "offline"}"></div>
              <div>
                <div class="hw-name">${escHtml(hw.name)}</div>
                <div style="font-size:11px;color:var(--text-muted);font-family:var(--font-mono);">${escHtml(hw.detail)}</div>
              </div>
              <div class="hw-status ${hw.online ? "online" : "offline"}">${hw.online ? "ONLINE" : "OFFLINE"}</div>
            </div>`)
          .join("");
      });
  }

  // ── System Pill ───────────────────────────────────────────────────────────
  function updateSystemPill(status, label) {
    if (!systemLabel || !systemPill) return;
    systemLabel.textContent = label || "SYSTEM NOMINAL";
    systemPill.className = "system-pill";
    if (status === "running") {
      systemPill.classList.add("warning");
    } else if (status === "degraded" || status === "error") {
      systemPill.classList.add("danger");
    }
  }

  // ── Modal: Add / Upload Experiment Script ─────────────────────────────────
  const modal = $("#modal-add-script");
  const scriptTextarea = $("#script-textarea");
  const scriptError = $("#script-error");
  const fileInput = $("#script-file-input");
  const fileUploadArea = $("#file-upload-area");
  const btnAddScript = $("#btn-add-script");
  const btnInsertTemplate = $("#btn-insert-template");

  if (btnAddScript) {
    btnAddScript.addEventListener("click", () => {
      modal.classList.add("visible");
      scriptTextarea.value = "";
      if (scriptError) scriptError.classList.remove("visible");
    });
  }

  $("#modal-close").addEventListener("click", closeModal);
  $("#btn-modal-cancel").addEventListener("click", closeModal);

  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  function closeModal() {
    modal.classList.remove("visible");
  }

  if (btnInsertTemplate) {
    btnInsertTemplate.addEventListener("click", () => {
      const template = {
        id: `phase-${experiments.length + 1}`,
        name: `Phase ${experiments.length + 1} — Custom Protocol`,
        steps: [
          { id: 1, label: "Inspect orange circular cap (5s hold)", expected_object: "orange_circular_cap", expected_action: "INSPECT", duration: 5.0 },
          { id: 2, label: "Place orange cap on yellow square", expected_object: "orange_circular_cap", expected_action: "RELEASE", target_surface: "yellow_square" }
        ]
      };
      scriptTextarea.value = JSON.stringify(template, null, 2);
    });
  }

  // File Upload
  if (fileUploadArea && fileInput) {
    fileUploadArea.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        scriptTextarea.value = ev.target.result;
      };
      reader.readAsText(file);
    });

    fileUploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      fileUploadArea.style.borderColor = "var(--accent)";
    });
    fileUploadArea.addEventListener("dragleave", () => {
      fileUploadArea.style.borderColor = "";
    });
    fileUploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      fileUploadArea.style.borderColor = "";
      const file = e.dataTransfer.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          scriptTextarea.value = ev.target.result;
        };
        reader.readAsText(file);
      }
    });
  }

  // Modal Submit
  $("#btn-modal-submit").addEventListener("click", () => {
    const raw = scriptTextarea.value.trim();
    if (!raw) {
      showScriptError("Please paste or upload a JSON script.");
      return;
    }
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      showScriptError("Invalid JSON syntax: " + e.message);
      return;
    }

    fetch("/api/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          showScriptError(data.error);
        } else {
          closeModal();
          showToast(`Protocol '${parsed.name || parsed.id}' added to catalog!`, "success");
          loadExperiments();
        }
      })
      .catch(() => showScriptError("Server communication error."));
  });

  function showScriptError(msg) {
    if (!scriptError) return;
    scriptError.textContent = msg;
    scriptError.classList.add("visible");
  }

  // ── Toast Notifications ───────────────────────────────────────────────────
  function showToast(message, type) {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = "toast " + (type || "");
    toast.innerHTML = `<i class="ph ${type === 'success' ? 'ph-check-circle' : type === 'danger' ? 'ph-warning-octagon' : 'ph-info'}"></i> <span>${escHtml(message)}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(40px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4500);
  }

  // ── Helper Utilities ──────────────────────────────────────────────────────
  function escHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  function truncate(str, len) {
    if (!str) return "";
    return str.length > len ? str.slice(0, len) + "…" : str;
  }

  function formatTime(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return iso;
    }
  }

  // Global exposure for inline onclicks
  window.VIGIL = { selectPhase };

  // ── Initialization ────────────────────────────────────────────────────────
  loadExperiments();
  loadAvailableCameras();

  // Request browser webcam permission immediately on load!
  startBrowserWebcam();
})();
