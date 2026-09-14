/**
 * canvas.js - High-DPI 2D Interactive Canvas Renderer with Voronoi Decision Boundaries & Animations
 */

export class CanvasRenderer {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.clusterColors = options.clusterColors || [
      '#38bdf8', '#f43f5e', '#10b981', '#a855f7', '#f59e0b',
      '#06b6d4', '#ec4899', '#84cc16', '#6366f1', '#f97316'
    ];

    // Rendering options
    this.showVoronoi = true;
    this.showTrails = true;
    this.showLines = false;
    this.showCentroids = true;
    this.voronoiResolution = 10; // Pixel step for voronoi shading

    // Centroid Animation State (for smooth 60fps interpolation)
    this.animatedCentroids = [];
    this.targetCentroids = [];
    this.animationProgress = 1.0;
    this.animSpeed = 0.18; // Lerp speed

    // Resize handling
    this.dpr = window.devicePixelRatio || 1;
    this.width = 800;
    this.height = 600;
    this.setupResize();

    // Mouse Interaction
    this.hoverPoint = null;
    this.hoverCentroid = null;
    this.isMouseDown = false;
    this.activeTool = 'brush'; // 'brush', 'manualCentroid', 'eraser', 'select'
    this.brushRadius = 35;
    this.brushDensity = 4;

    this.onCanvasInteraction = options.onCanvasInteraction || null;
    this.setupEvents();
  }

  setupResize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = Math.max(300, rect.width || 800);
    this.height = Math.max(300, rect.height || 600);

    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.resetTransform();
    this.ctx.scale(this.dpr, this.dpr);
  }

  setupEvents() {
    window.addEventListener('resize', () => {
      this.setupResize();
    });

    const getPos = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left),
        y: (e.clientY - rect.top)
      };
    };

    this.canvas.addEventListener('mousedown', (e) => {
      this.isMouseDown = true;
      const pos = getPos(e);
      if (this.onCanvasInteraction) {
        this.onCanvasInteraction('down', pos, this.activeTool);
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (this.isMouseDown) {
        this.isMouseDown = false;
        const pos = getPos(e);
        if (this.onCanvasInteraction) {
          this.onCanvasInteraction('up', pos, this.activeTool);
        }
      }
    });

    this.canvas.addEventListener('mousemove', (e) => {
      const pos = getPos(e);
      if (this.isMouseDown) {
        if (this.onCanvasInteraction) {
          this.onCanvasInteraction('drag', pos, this.activeTool);
        }
      } else {
        if (this.onCanvasInteraction) {
          this.onCanvasInteraction('hover', pos, this.activeTool);
        }
      }
    });
  }

  setCentroidTargets(centroids) {
    this.targetCentroids = centroids.map(c => ({ x: c.x, y: c.y }));
    if (this.animatedCentroids.length !== centroids.length) {
      this.animatedCentroids = centroids.map(c => ({ x: c.x, y: c.y }));
    }
  }

  updateAnimations() {
    if (this.targetCentroids.length === 0) return;
    for (let i = 0; i < this.targetCentroids.length; i++) {
      if (!this.animatedCentroids[i]) {
        this.animatedCentroids[i] = { ...this.targetCentroids[i] };
      } else {
        this.animatedCentroids[i].x += (this.targetCentroids[i].x - this.animatedCentroids[i].x) * this.animSpeed;
        this.animatedCentroids[i].y += (this.targetCentroids[i].y - this.animatedCentroids[i].y) * this.animSpeed;
      }
    }
  }

  render(engine, customHover = null) {
    this.updateAnimations();

    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    // Draw Subtle Background Grid
    this.drawGrid();

    const centroidsToUse = this.animatedCentroids.length === engine.centroids.length && this.animatedCentroids.length > 0
      ? this.animatedCentroids
      : engine.centroids;

    // 1. Draw Voronoi Decision Boundaries
    if (this.showVoronoi && centroidsToUse.length > 0) {
      this.drawVoronoiBoundaries(centroidsToUse, engine.distanceFn);
    }

    // 2. Draw Assignment Distance Lines
    if (this.showLines && centroidsToUse.length > 0) {
      this.drawAssignmentLines(engine.points, engine.assignments, centroidsToUse);
    }

    // 3. Draw Centroid Trajectory Trails
    if (this.showTrails && engine.centroidTrajectories) {
      this.drawTrajectoryTrails(engine.centroidTrajectories);
    }

    // 4. Draw Data Points
    this.drawPoints(engine.points, engine.assignments);

    // 5. Draw Centroids
    if (this.showCentroids && centroidsToUse.length > 0) {
      this.drawCentroids(centroidsToUse);
    }

    // 6. Draw Hover / Selection Info
    if (customHover) {
      this.drawHoverTooltip(customHover);
    }
  }

  drawGrid() {
    const ctx = this.ctx;
    const gridSize = 40;
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.05)';
    ctx.lineWidth = 1;

    ctx.beginPath();
    for (let x = 0; x < this.width; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.height);
    }
    for (let y = 0; y < this.height; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(this.width, y);
    }
    ctx.stroke();
  }

  drawVoronoiBoundaries(centroids, distanceFn) {
    const ctx = this.ctx;
    const step = this.voronoiResolution;
    const cols = Math.ceil(this.width / step);
    const rows = Math.ceil(this.height / step);

    // Pre-calculate RGB values for faster opacity painting
    const parsedColors = this.clusterColors.map(hex => {
      const c = parseInt(hex.replace('#', ''), 16);
      return {
        r: (c >> 16) & 255,
        g: (c >> 8) & 255,
        b: c & 255
      };
    });

    for (let x = 0; x < cols; x++) {
      for (let y = 0; y < rows; y++) {
        const px = x * step + step / 2;
        const py = y * step + step / 2;

        let minDist = Infinity;
        let closest = 0;

        for (let c = 0; c < centroids.length; c++) {
          const d = distanceFn({ x: px, y: py }, centroids[c]);
          if (d < minDist) {
            minDist = d;
            closest = c;
          }
        }

        const color = parsedColors[closest % parsedColors.length];
        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.08)`;
        ctx.fillRect(x * step, y * step, step, step);
      }
    }
  }

  drawAssignmentLines(points, assignments, centroids) {
    const ctx = this.ctx;
    ctx.lineWidth = 0.6;
    ctx.setLineDash([3, 4]);

    for (let i = 0; i < points.length; i++) {
      const cIdx = assignments[i];
      if (cIdx >= 0 && cIdx < centroids.length) {
        const c = centroids[cIdx];
        const color = this.clusterColors[cIdx % this.clusterColors.length];
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.25;

        ctx.beginPath();
        ctx.moveTo(points[i].x, points[i].y);
        ctx.lineTo(c.x, c.y);
        ctx.stroke();
      }
    }

    ctx.setLineDash([]);
    ctx.globalAlpha = 1.0;
  }

  drawTrajectoryTrails(trajectories) {
    const ctx = this.ctx;
    trajectories.forEach((trail, cIdx) => {
      if (trail.length <= 1) return;
      const color = this.clusterColors[cIdx % this.clusterColors.length];

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([2, 2]);

      ctx.beginPath();
      ctx.moveTo(trail[0].x, trail[0].y);
      for (let i = 1; i < trail.length; i++) {
        ctx.lineTo(trail[i].x, trail[i].y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw small trail dots
      trail.forEach((pt, idx) => {
        if (idx < trail.length - 1) {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }
      });
    });
  }

  drawPoints(points, assignments) {
    const ctx = this.ctx;
    const ptRadius = 5.5;

    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      const cIdx = assignments[i];
      const color = (cIdx >= 0 && cIdx < this.clusterColors.length)
        ? this.clusterColors[cIdx]
        : '#94a3b8';

      // Outer point glow
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, ptRadius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = cIdx >= 0 ? 8 : 0;
      ctx.fill();

      // Inner stroke
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#090d16';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Draw State Name Badge if present
      if (pt.meta && pt.meta.state) {
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 10px Outfit, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        
        // Subtle text backdrop glow
        ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
        ctx.shadowBlur = 4;
        ctx.fillText(pt.meta.state, pt.x + 9, pt.y);
        ctx.shadowBlur = 0;
      }
    }
  }

  drawCentroids(centroids) {
    const ctx = this.ctx;

    centroids.forEach((c, idx) => {
      const color = this.clusterColors[idx % this.clusterColors.length];

      // 1. Outer Pulse Aura
      ctx.beginPath();
      ctx.arc(c.x, c.y, 20, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.15;
      ctx.fill();
      ctx.globalAlpha = 1.0;

      // 2. Halo ring
      ctx.beginPath();
      ctx.arc(c.x, c.y, 14, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // 3. Center Solid Core
      ctx.beginPath();
      ctx.arc(c.x, c.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // 4. Centroid Index Label
      ctx.fillStyle = '#0f172a';
      ctx.font = 'bold 9px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`μ${idx + 1}`, c.x, c.y + 0.5);

      // Centroid Floating Tag
      ctx.fillStyle = color;
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      ctx.fillText(`C${idx + 1}`, c.x, c.y - 20);
    });
  }

  drawHoverTooltip(hover) {
    const ctx = this.ctx;
    const { x, y, lines } = hover;
    const lineList = Array.isArray(lines) ? lines : [hover.text || ''];

    const padX = 12, padY = 8;
    ctx.font = '11px JetBrains Mono, monospace';

    let maxW = 0;
    lineList.forEach(l => {
      const w = ctx.measureText(l).width;
      if (w > maxW) maxW = w;
    });

    const boxW = maxW + padX * 2;
    const lineHeight = 16;
    const boxH = lineList.length * lineHeight + padY * 2;

    const boxX = Math.min(this.width - boxW - 10, Math.max(10, x + 14));
    const boxY = Math.min(this.height - boxH - 10, Math.max(10, y - boxH / 2));

    // Backdrop
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 8);
    ctx.fill();
    ctx.stroke();

    // Text Lines
    lineList.forEach((line, idx) => {
      ctx.fillStyle = idx === 0 ? '#38bdf8' : '#cbd5e1';
      ctx.font = idx === 0 ? 'bold 11px Outfit, sans-serif' : '10px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(line, boxX + padX, boxY + padY + idx * lineHeight);
    });
  }
}
