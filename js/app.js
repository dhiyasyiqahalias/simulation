/**
 * app.js - Main Application Orchestrator for Interactive K-Means Clustering Simulation
 */

import { KMeansEngine } from './kmeans.js';
import { DatasetGenerators } from './datasets.js';
import { CanvasRenderer } from './canvas.js';
import { MetricsCalculator } from './metrics.js';
import { ChartRenderer } from './chart-renderer.js';
import { ChatAssistant } from './chat-assistant.js';
import { projectTourismPoints, MalaysianTourismData, TourismDimensions } from './tourism-data.js';

class KMeansApp {
  constructor() {
    this.clusterColors = [
      '#38bdf8', '#f43f5e', '#10b981', '#a855f7', '#f59e0b',
      '#06b6d4', '#ec4899', '#84cc16', '#6366f1', '#f97316'
    ];

    // Engine & Renderer
    this.engine = new KMeansEngine({ k: 3, metric: 'euclidean', initStrategy: 'kmeans++' });
    this.canvasEl = document.getElementById('simulationCanvas');
    this.renderer = new CanvasRenderer(this.canvasEl, {
      clusterColors: this.clusterColors,
      onCanvasInteraction: this.handleCanvasInteraction.bind(this)
    });

    // Playback state
    this.isPlaying = false;
    this.playTimer = null;
    this.playSpeedMs = 500; // ms per step

    // Active Dataset Config (Default to Sustainable Tourism from Excel!)
    this.currentPreset = 'tourism';
    this.tourismXDim = 'density';
    this.tourismYDim = 'spend';
    this.pointCount = 300;
    this.manualCentroidsToPlace = [];

    // Analytics state
    this.activeAnalyticsTab = 'inertia'; // 'inertia', 'silhouette', 'elbow'
    this.silhouetteData = null;
    this.elbowData = null;
    this.hoverInfo = null;

    this.initDOMReferences();
    this.bindEvents();
    this.chatAssistant = new ChatAssistant(this);
    this.generateDataset(this.currentPreset);
    this.startAnimationLoop();
  }

  initDOMReferences() {
    // Buttons
    this.btnPlay = document.getElementById('btnPlay');
    this.btnStepForward = document.getElementById('btnStepForward');
    this.btnStepBack = document.getElementById('btnStepBack');
    this.btnRunConverge = document.getElementById('btnRunConverge');
    this.btnResetCentroids = document.getElementById('btnResetCentroids');
    this.btnNewData = document.getElementById('btnNewData');
    this.btnClearCanvas = document.getElementById('btnClearCanvas');
    this.btnRunElbow = document.getElementById('btnRunElbow');
    this.btnInfo = document.getElementById('btnInfo');
    this.btnCloseModal = document.getElementById('btnCloseModal');
    this.modalHelp = document.getElementById('modalHelpHolder');

    // Controls
    this.sliderK = document.getElementById('sliderK');
    this.valK = document.getElementById('valK');
    this.selectMetric = document.getElementById('selectMetric');
    this.selectInit = document.getElementById('selectInit');
    this.sliderSpeed = document.getElementById('sliderSpeed');
    this.valSpeed = document.getElementById('valSpeed');
    this.sliderPoints = document.getElementById('sliderPoints');
    this.valPoints = document.getElementById('valPoints');

    // Tourism Specific Controls
    this.tourismDimensionsCard = document.getElementById('tourismDimensionsCard');
    this.pointCountContainer = document.getElementById('pointCountContainer');
    this.selectTourismX = document.getElementById('selectTourismX');
    this.selectTourismY = document.getElementById('selectTourismY');

    // Toggles
    this.toggleVoronoi = document.getElementById('toggleVoronoi');
    this.toggleTrails = document.getElementById('toggleTrails');
    this.toggleLines = document.getElementById('toggleLines');

    // HUD & Stats
    this.hudIteration = document.getElementById('hudIteration');
    this.hudPhase = document.getElementById('hudPhase');
    this.hudInertia = document.getElementById('hudInertia');
    this.hudSilhouette = document.getElementById('hudSilhouette');
    this.hudPoints = document.getElementById('hudPoints');
    this.statusBadge = document.getElementById('statusBadge');
    this.clusterTableBody = document.getElementById('clusterTableBody');

    // Charts
    this.inertiaChartCanvas = document.getElementById('inertiaChartCanvas');
    this.silhouetteChartCanvas = document.getElementById('silhouetteChartCanvas');
    this.elbowChartCanvas = document.getElementById('elbowChartCanvas');
  }

  bindEvents() {
    // Playback
    this.btnPlay.addEventListener('click', () => this.togglePlay());
    this.btnStepForward.addEventListener('click', () => this.stepForward());
    this.btnStepBack.addEventListener('click', () => this.stepBack());
    this.btnRunConverge.addEventListener('click', () => this.runToConvergence());
    this.btnResetCentroids.addEventListener('click', () => this.resetCentroids());
    this.btnNewData.addEventListener('click', () => this.generateDataset(this.currentPreset));
    this.btnClearCanvas.addEventListener('click', () => this.clearAllPoints());

    // Settings
    this.sliderK.addEventListener('input', (e) => {
      const k = parseInt(e.target.value);
      this.valK.textContent = k;
      this.engine.setK(k);
      this.resetCentroids();
    });

    this.selectMetric.addEventListener('change', (e) => {
      this.engine.setMetric(e.target.value);
      this.updateAnalytics();
    });

    this.selectInit.addEventListener('change', (e) => {
      this.engine.setInitStrategy(e.target.value);
      if (e.target.value === 'manual') {
        this.setTool('manualCentroid');
      }
    });

    this.sliderSpeed.addEventListener('input', (e) => {
      const spd = parseFloat(e.target.value);
      this.valSpeed.textContent = `${spd}x`;
      this.playSpeedMs = Math.round(500 / spd);
      if (this.isPlaying) {
        this.pause();
        this.play();
      }
    });

    this.sliderPoints.addEventListener('input', (e) => {
      this.pointCount = parseInt(e.target.value);
      this.valPoints.textContent = this.pointCount;
    });

    this.sliderPoints.addEventListener('change', () => {
      this.generateDataset(this.currentPreset);
    });

    // Tourism Dimension Selectors
    this.selectTourismX?.addEventListener('change', (e) => {
      this.tourismXDim = e.target.value;
      if (this.currentPreset === 'tourism') {
        this.generateDataset('tourism');
      }
    });

    this.selectTourismY?.addEventListener('change', (e) => {
      this.tourismYDim = e.target.value;
      if (this.currentPreset === 'tourism') {
        this.generateDataset('tourism');
      }
    });

    // Preset Buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentPreset = btn.dataset.preset;
        this.generateDataset(this.currentPreset);
      });
    });

    // Tool Buttons
    document.querySelectorAll('.tool-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.setTool(btn.dataset.tool);
      });
    });

    // Display Toggles
    this.toggleVoronoi.addEventListener('change', (e) => {
      this.renderer.showVoronoi = e.target.checked;
    });
    this.toggleTrails.addEventListener('change', (e) => {
      this.renderer.showTrails = e.target.checked;
    });
    this.toggleLines.addEventListener('change', (e) => {
      this.renderer.showLines = e.target.checked;
    });

    // Analytics Tabs
    document.querySelectorAll('.tab-btn').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        this.switchAnalyticsTab(tab.dataset.tab);
      });
    });

    this.btnRunElbow.addEventListener('click', () => {
      this.runElbowAnalysis();
    });

    // Info Modal
    this.btnInfo.addEventListener('click', () => {
      this.modalHelp.classList.add('open');
    });
    this.btnCloseModal.addEventListener('click', () => {
      this.modalHelp.classList.remove('open');
    });
    this.modalHelp.addEventListener('click', (e) => {
      if (e.target === this.modalHelp) {
        this.modalHelp.classList.remove('open');
      }
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        this.togglePlay();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        this.stepForward();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        this.stepBack();
      } else if (e.code === 'KeyR') {
        this.resetCentroids();
      } else if (e.code === 'KeyG') {
        this.generateDataset(this.currentPreset);
      }
    });

    // CSV Import / Export
    document.getElementById('btnExportCSV')?.addEventListener('click', () => this.exportCSV());
    document.getElementById('fileImportCSV')?.addEventListener('change', (e) => this.importCSV(e));
  }

  setTool(tool) {
    this.renderer.activeTool = tool;
    document.querySelectorAll('.tool-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tool === tool);
    });
  }

  generateDataset(preset) {
    this.pause();
    const w = this.renderer.width || 800;
    const h = this.renderer.height || 600;
    const k = this.engine.k;

    // Toggle Tourism Controls
    if (preset === 'tourism') {
      if (this.tourismDimensionsCard) this.tourismDimensionsCard.style.display = 'flex';
      if (this.pointCountContainer) this.pointCountContainer.style.display = 'none';
      const points = projectTourismPoints(this.tourismXDim, this.tourismYDim, w, h, 60);
      this.engine.setPoints(points);
    } else {
      if (this.tourismDimensionsCard) this.tourismDimensionsCard.style.display = 'none';
      if (this.pointCountContainer) this.pointCountContainer.style.display = 'flex';

      let points = [];
      if (preset === 'blobs') {
        points = DatasetGenerators.blobs(this.pointCount, k, w, h);
      } else if (preset === 'concentric') {
        points = DatasetGenerators.concentricRings(this.pointCount, 3, w, h);
      } else if (preset === 'anisotropic') {
        points = DatasetGenerators.anisotropic(this.pointCount, w, h);
      } else if (preset === 'moons') {
        points = DatasetGenerators.moons(this.pointCount, w, h);
      } else if (preset === 'variedDensity') {
        points = DatasetGenerators.variedDensity(this.pointCount, w, h);
      } else if (preset === 'smiley') {
        points = DatasetGenerators.smiley(this.pointCount, w, h);
      } else {
        points = DatasetGenerators.uniform(this.pointCount, w, h);
      }
      this.engine.setPoints(points);
    }

    this.resetCentroids();
  }

  clearAllPoints() {
    this.pause();
    this.engine.setPoints([]);
    this.resetCentroids();
  }

  resetCentroids() {
    this.pause();
    const bounds = { width: this.renderer.width, height: this.renderer.height };
    this.engine.initializeCentroids(null, bounds);
    this.renderer.setCentroidTargets(this.engine.centroids);
    this.updateUI();
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    if (this.engine.isConverged) {
      this.resetCentroids();
    }
    this.isPlaying = true;
    this.btnPlay.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
    this.btnPlay.setAttribute('data-tooltip', 'Pause (Space)');

    const loop = () => {
      if (!this.isPlaying) return;
      if (this.engine.isConverged) {
        this.pause();
        return;
      }
      this.stepForward();
      this.playTimer = setTimeout(loop, this.playSpeedMs);
    };
    this.playTimer = setTimeout(loop, this.playSpeedMs);
  }

  pause() {
    this.isPlaying = false;
    clearTimeout(this.playTimer);
    this.btnPlay.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    this.btnPlay.setAttribute('data-tooltip', 'Play (Space)');
  }

  stepForward() {
    if (this.engine.isConverged) return;
    this.engine.step();
    this.renderer.setCentroidTargets(this.engine.centroids);
    this.updateUI();
  }

  stepBack() {
    this.pause();
    const success = this.engine.stepBack();
    if (success) {
      this.renderer.setCentroidTargets(this.engine.centroids);
      this.updateUI();
    }
  }

  runToConvergence() {
    this.pause();
    this.engine.runToConvergence();
    this.renderer.setCentroidTargets(this.engine.centroids);
    this.updateUI();
  }

  runElbowAnalysis() {
    if (this.engine.points.length === 0) return;
    this.btnRunElbow.disabled = true;
    this.btnRunElbow.textContent = 'Computing...';

    setTimeout(() => {
      const bounds = { width: this.renderer.width, height: this.renderer.height };
      this.elbowData = MetricsCalculator.computeElbowCurve(this.engine.points, 8, 4, bounds);
      this.btnRunElbow.disabled = false;
      this.btnRunElbow.textContent = 'Run Elbow Sweep';
      this.switchAnalyticsTab('elbow');
    }, 50);
  }

  switchAnalyticsTab(tab) {
    this.activeAnalyticsTab = tab;
    document.getElementById('inertiaCard').style.display = tab === 'inertia' ? 'flex' : 'none';
    document.getElementById('silhouetteCard').style.display = tab === 'silhouette' ? 'flex' : 'none';
    document.getElementById('elbowCard').style.display = tab === 'elbow' ? 'flex' : 'none';
    this.updateAnalytics();
  }

  updateUI() {
    // HUD Stats
    this.hudIteration.textContent = this.engine.iteration;
    this.hudPhase.textContent = this.engine.phase;
    this.hudPoints.textContent = this.engine.points.length;

    const wcss = this.engine.calculateWCSS();
    this.hudInertia.textContent = wcss >= 1000000 ? `${(wcss / 1000000).toFixed(2)}M` : wcss >= 1000 ? `${(wcss / 1000).toFixed(1)}k` : Math.round(wcss);

    // Status Badge
    if (this.engine.isConverged) {
      this.statusBadge.className = 'status-pill status-converged';
      this.statusBadge.innerHTML = `<span class="pulse-dot"></span> Converged`;
    } else if (this.isPlaying || this.engine.iteration > 0) {
      this.statusBadge.className = 'status-pill status-running';
      this.statusBadge.innerHTML = `<span class="pulse-dot"></span> Iterating (${this.engine.phase})`;
    } else {
      this.statusBadge.className = 'status-pill status-init';
      this.statusBadge.innerHTML = `<span class="pulse-dot"></span> Ready`;
    }

    this.updateAnalytics();
    this.updateClusterTable();
  }

  updateAnalytics() {
    // Update Silhouette
    if (this.engine.centroids.length >= 2 && this.engine.phase !== 'IDLE') {
      this.silhouetteData = MetricsCalculator.calculateSilhouette(
        this.engine.points,
        this.engine.assignments,
        this.engine.centroids.length
      );
      this.hudSilhouette.textContent = this.silhouetteData.overall.toFixed(3);
    } else {
      this.hudSilhouette.textContent = '--';
    }

    // Render active charts
    if (this.activeAnalyticsTab === 'inertia') {
      ChartRenderer.renderInertiaChart(this.inertiaChartCanvas, this.engine.history);
    } else if (this.activeAnalyticsTab === 'silhouette') {
      ChartRenderer.renderSilhouetteChart(this.silhouetteChartCanvas, this.silhouetteData, this.clusterColors);
    } else if (this.activeAnalyticsTab === 'elbow') {
      ChartRenderer.renderElbowChart(this.elbowChartCanvas, this.elbowData);
    }
  }

  updateClusterTable() {
    if (!this.clusterTableBody) return;
    const stats = MetricsCalculator.getDiagnostics(
      this.engine.points,
      this.engine.centroids,
      this.engine.assignments
    );

    const isTourism = this.currentPreset === 'tourism';

    this.clusterTableBody.innerHTML = stats.map(st => {
      const color = this.clusterColors[st.cluster % this.clusterColors.length];
      
      // Collect State Names in this cluster if Tourism dataset
      let statesListHtml = '';
      if (isTourism) {
        const memberStates = [];
        this.engine.points.forEach((p, idx) => {
          if (this.engine.assignments[idx] === st.cluster && p.meta) {
            memberStates.push(`<span style="background: rgba(148, 163, 184, 0.15); padding: 1px 4px; border-radius: 3px; font-size: 0.68rem; margin: 1px; display: inline-block;">${p.meta.state}</span>`);
          }
        });
        if (memberStates.length > 0) {
          statesListHtml = `<div style="margin-top: 4px; font-size: 0.68rem; line-height: 1.3;">${memberStates.join(' ')}</div>`;
        }
      }

      return `
        <tr>
          <td>
            <div style="display: flex; align-items: center;">
              <span class="cluster-badge-dot" style="background:${color}"></span>
              <strong>Cluster ${st.cluster + 1}</strong>
            </div>
            ${statesListHtml}
          </td>
          <td>(${st.centroid.x}, ${st.centroid.y})</td>
          <td><strong>${st.count}</strong> (${st.percentage}%)</td>
          <td>±${st.stdDev}px</td>
        </tr>
      `;
    }).join('');
  }

  handleCanvasInteraction(type, pos, tool) {
    if (tool === 'brush') {
      if (type === 'down' || type === 'drag') {
        const numNew = type === 'down' ? 6 : 2;
        const pts = [];
        for (let i = 0; i < numNew; i++) {
          const u = 1 - Math.random();
          const v = Math.random();
          const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
          pts.push({
            x: Math.max(10, Math.min(this.renderer.width - 10, pos.x + z * 18)),
            y: Math.max(10, Math.min(this.renderer.height - 10, pos.y + (Math.random() - 0.5) * 36)),
            trueCluster: -1
          });
        }
        this.engine.points.push(...pts);
        this.engine.assignments.push(...new Array(pts.length).fill(-1));
        if (this.engine.centroids.length > 0) {
          this.engine.recomputeAssignments();
        }
        this.updateUI();
      }
    } else if (tool === 'eraser') {
      if (type === 'down' || type === 'drag') {
        const eraseRadius = 30;
        const keptPoints = [];
        const keptAssignments = [];
        for (let i = 0; i < this.engine.points.length; i++) {
          const pt = this.engine.points[i];
          if (Math.hypot(pt.x - pos.x, pt.y - pos.y) > eraseRadius) {
            keptPoints.push(pt);
            keptAssignments.push(this.engine.assignments[i]);
          }
        }
        if (keptPoints.length !== this.engine.points.length) {
          this.engine.points = keptPoints;
          this.engine.assignments = keptAssignments;
          if (this.engine.centroids.length > 0) {
            this.engine.recomputeAssignments();
          }
          this.updateUI();
        }
      }
    } else if (tool === 'manualCentroid') {
      if (type === 'down') {
        if (this.manualCentroidsToPlace.length >= this.engine.k) {
          this.manualCentroidsToPlace = [];
        }
        this.manualCentroidsToPlace.push({ x: pos.x, y: pos.y });

        if (this.manualCentroidsToPlace.length === this.engine.k) {
          this.engine.initializeCentroids(this.manualCentroidsToPlace);
          this.renderer.setCentroidTargets(this.engine.centroids);
          this.manualCentroidsToPlace = [];
          this.setTool('brush');
          this.updateUI();
        }
      }
    } else if (type === 'hover') {
      // Check if hovering a centroid or point for tooltip inspection
      let found = null;
      for (let c = 0; c < this.engine.centroids.length; c++) {
        const cent = this.engine.centroids[c];
        if (Math.hypot(cent.x - pos.x, cent.y - pos.y) < 18) {
          found = {
            x: cent.x,
            y: cent.y,
            lines: [`Centroid μ${c + 1}: (${Math.round(cent.x)}, ${Math.round(cent.y)})`, `Cluster Center ${c + 1}`]
          };
          break;
        }
      }

      if (!found) {
        for (let i = 0; i < this.engine.points.length; i++) {
          const pt = this.engine.points[i];
          if (Math.hypot(pt.x - pos.x, pt.y - pos.y) < 12) {
            const cIdx = this.engine.assignments[i];
            const clusterText = cIdx >= 0 ? `Cluster ${cIdx + 1}` : 'Unassigned';

            if (pt.meta && pt.meta.state) {
              const xDimInfo = TourismDimensions[this.tourismXDim] || { label: 'X', format: (v) => v };
              const yDimInfo = TourismDimensions[this.tourismYDim] || { label: 'Y', format: (v) => v };

              found = {
                x: pt.x,
                y: pt.y,
                lines: [
                  `🇲🇾 ${pt.meta.state} (Tier: ${pt.meta.tier})`,
                  `Assigned to: ${clusterText}`,
                  `• ${xDimInfo.label}: ${xDimInfo.format(pt.meta[this.tourismXDim])}`,
                  `• ${yDimInfo.label}: ${yDimInfo.format(pt.meta[this.tourismYDim])}`,
                  `• Domestic Visitors: ${pt.meta.visitors.toLocaleString()}k`,
                  `• Population: ${pt.meta.population.toLocaleString()}k`,
                  `• Total Receipts: RM ${pt.meta.receipts.toLocaleString()}M`
                ]
              };
            } else {
              found = {
                x: pt.x,
                y: pt.y,
                lines: [`Point (${Math.round(pt.x)}, ${Math.round(pt.y)})`, clusterText]
              };
            }
            break;
          }
        }
      }
      this.hoverInfo = found;
    }
  }

  exportCSV() {
    if (this.engine.points.length === 0) return;
    let csv = 'x,y,cluster,trueCluster\n';
    this.engine.points.forEach((p, idx) => {
      csv += `${p.x.toFixed(2)},${p.y.toFixed(2)},${this.engine.assignments[idx]},${p.trueCluster}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `kmeans_dataset_k${this.engine.k}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  importCSV(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.trim().split('\n');
      const pts = [];
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(',');
        if (parts.length >= 2) {
          const x = parseFloat(parts[0]);
          const y = parseFloat(parts[1]);
          const trueCluster = parts.length >= 4 ? parseInt(parts[3]) : -1;
          if (!isNaN(x) && !isNaN(y)) {
            pts.push({ x, y, trueCluster });
          }
        }
      }
      if (pts.length > 0) {
        this.engine.setPoints(pts);
        this.resetCentroids();
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  }

  startAnimationLoop() {
    const renderLoop = () => {
      this.renderer.render(this.engine, this.hoverInfo);
      requestAnimationFrame(renderLoop);
    };
    requestAnimationFrame(renderLoop);
  }
}

// Bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.kmeansApp = new KMeansApp();
});
