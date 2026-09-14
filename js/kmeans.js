/**
 * kmeans.js - K-Means Clustering Core Engine with State Machine and Time-Travel History
 */

export const DistanceMetrics = {
  euclidean: (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y),
  squaredEuclidean: (p1, p2) => (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2,
  manhattan: (p1, p2) => Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y),
  chebyshev: (p1, p2) => Math.max(Math.abs(p1.x - p2.x), Math.abs(p1.y - p2.y))
};

export class KMeansEngine {
  constructor(options = {}) {
    this.k = options.k || 3;
    this.metricName = options.metric || 'euclidean';
    this.initStrategy = options.initStrategy || 'kmeans++';
    this.tolerance = options.tolerance || 0.001;
    this.maxIterations = options.maxIterations || 100;

    this.points = [];
    this.centroids = [];
    this.assignments = [];
    this.centroidTrajectories = []; // Trajectory history for animation trails

    this.iteration = 0;
    this.phase = 'IDLE'; // 'IDLE', 'INITIALIZED', 'ASSIGNMENT', 'UPDATE', 'CONVERGED'
    this.isConverged = false;
    this.history = [];
    this.historyIndex = -1;
  }

  get distanceFn() {
    return DistanceMetrics[this.metricName] || DistanceMetrics.euclidean;
  }

  setPoints(points) {
    this.points = points.map(p => ({ x: p.x, y: p.y, trueCluster: p.trueCluster, meta: p.meta }));
    this.reset();
  }

  setK(k) {
    this.k = Math.max(1, Math.min(10, k));
    this.reset();
  }

  setMetric(metricName) {
    this.metricName = metricName;
    if (this.phase !== 'IDLE') {
      this.recomputeAssignments();
    }
  }

  setInitStrategy(strategy) {
    this.initStrategy = strategy;
  }

  reset() {
    this.centroids = [];
    this.assignments = new Array(this.points.length).fill(-1);
    this.centroidTrajectories = [];
    this.iteration = 0;
    this.phase = 'IDLE';
    this.isConverged = false;
    this.history = [];
    this.historyIndex = -1;
  }

  /**
   * Initialize Centroids based on strategy
   */
  initializeCentroids(manualCentroids = null, bounds = { width: 800, height: 600 }) {
    if (this.points.length === 0) return false;
    const k = Math.min(this.k, this.points.length);
    this.centroids = [];
    this.centroidTrajectories = Array.from({ length: k }, () => []);

    if (manualCentroids && manualCentroids.length === k) {
      this.centroids = manualCentroids.map(c => ({ x: c.x, y: c.y }));
    } else if (this.initStrategy === 'kmeans++') {
      this.centroids = this._initKMeansPlusPlus(k);
    } else if (this.initStrategy === 'randomPoints') {
      this.centroids = this._initRandomPoints(k);
    } else {
      // randomUniform in bounding box
      this.centroids = this._initRandomUniform(k, bounds);
    }

    // Record initial trajectories
    this.centroids.forEach((c, idx) => {
      this.centroidTrajectories[idx].push({ x: c.x, y: c.y });
    });

    this.assignments = new Array(this.points.length).fill(-1);
    this.iteration = 0;
    this.phase = 'INITIALIZED';
    this.isConverged = false;
    this.history = [];
    this.historyIndex = -1;

    this.saveStateSnapshot('Initial Centroid Placement');
    return true;
  }

  _initRandomPoints(k) {
    const centroids = [];
    const usedIndices = new Set();
    while (centroids.length < k && usedIndices.size < this.points.length) {
      const idx = Math.floor(Math.random() * this.points.length);
      if (!usedIndices.has(idx)) {
        usedIndices.add(idx);
        centroids.push({ x: this.points[idx].x, y: this.points[idx].y });
      }
    }
    return centroids;
  }

  _initRandomUniform(k, bounds) {
    const centroids = [];
    const margin = 40;
    const minX = margin;
    const maxX = (bounds.width || 800) - margin;
    const minY = margin;
    const maxY = (bounds.height || 600) - margin;

    for (let i = 0; i < k; i++) {
      centroids.push({
        x: minX + Math.random() * (maxX - minX),
        y: minY + Math.random() * (maxY - minY)
      });
    }
    return centroids;
  }

  _initKMeansPlusPlus(k) {
    const centroids = [];
    // 1. Choose first center uniformly at random from data points
    const firstIdx = Math.floor(Math.random() * this.points.length);
    centroids.push({ x: this.points[firstIdx].x, y: this.points[firstIdx].y });

    // 2. Choose remaining centers with probability proportional to D(x)^2
    while (centroids.length < k) {
      const distSq = new Float64Array(this.points.length);
      let totalDistSq = 0;

      for (let i = 0; i < this.points.length; i++) {
        let minDistSq = Infinity;
        for (let c = 0; c < centroids.length; c++) {
          const d2 = (this.points[i].x - centroids[c].x) ** 2 + (this.points[i].y - centroids[c].y) ** 2;
          if (d2 < minDistSq) minDistSq = d2;
        }
        distSq[i] = minDistSq;
        totalDistSq += minDistSq;
      }

      if (totalDistSq === 0) {
        // Fallback to random point
        const fallbackIdx = Math.floor(Math.random() * this.points.length);
        centroids.push({ x: this.points[fallbackIdx].x, y: this.points[fallbackIdx].y });
        continue;
      }

      const r = Math.random() * totalDistSq;
      let cumSum = 0;
      let chosenIdx = 0;

      for (let i = 0; i < this.points.length; i++) {
        cumSum += distSq[i];
        if (cumSum >= r) {
          chosenIdx = i;
          break;
        }
      }

      centroids.push({ x: this.points[chosenIdx].x, y: this.points[chosenIdx].y });
    }

    return centroids;
  }

  /**
   * Step 1: Assign each point to its nearest centroid
   */
  assignPoints() {
    if (this.centroids.length === 0) return false;
    let changedCount = 0;

    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      let minDistance = Infinity;
      let closestCluster = 0;

      for (let c = 0; c < this.centroids.length; c++) {
        const dist = this.distanceFn(p, this.centroids[c]);
        if (dist < minDistance) {
          minDistance = dist;
          closestCluster = c;
        }
      }

      if (this.assignments[i] !== closestCluster) {
        changedCount++;
        this.assignments[i] = closestCluster;
      }
    }

    this.phase = 'ASSIGNMENT';
    this.saveStateSnapshot(`Assigned Points (Changes: ${changedCount})`);
    return changedCount;
  }

  /**
   * Step 2: Recompute centroid positions as mean of points
   */
  updateCentroids() {
    if (this.centroids.length === 0) return false;

    const k = this.centroids.length;
    const clusterSums = Array.from({ length: k }, () => ({ x: 0, y: 0, count: 0 }));

    for (let i = 0; i < this.points.length; i++) {
      const cluster = this.assignments[i];
      if (cluster >= 0 && cluster < k) {
        clusterSums[cluster].x += this.points[i].x;
        clusterSums[cluster].y += this.points[i].y;
        clusterSums[cluster].count++;
      }
    }

    let maxDisplacement = 0;
    const oldCentroids = this.centroids.map(c => ({ x: c.x, y: c.y }));

    for (let c = 0; c < k; c++) {
      if (clusterSums[c].count > 0) {
        const newX = clusterSums[c].x / clusterSums[c].count;
        const newY = clusterSums[c].y / clusterSums[c].count;
        const disp = Math.hypot(newX - oldCentroids[c].x, newY - oldCentroids[c].y);
        if (disp > maxDisplacement) maxDisplacement = disp;

        this.centroids[c].x = newX;
        this.centroids[c].y = newY;
      } else {
        // Handle empty cluster: re-seed centroid at random point furthest from its own center
        let furthestPoint = this.points[Math.floor(Math.random() * this.points.length)];
        this.centroids[c].x = furthestPoint.x;
        this.centroids[c].y = furthestPoint.y;
      }

      this.centroidTrajectories[c].push({ x: this.centroids[c].x, y: this.centroids[c].y });
    }

    this.iteration++;
    this.phase = 'UPDATE';

    // Convergence check
    if (maxDisplacement <= this.tolerance || this.iteration >= this.maxIterations) {
      this.isConverged = true;
      this.phase = 'CONVERGED';
    }

    this.saveStateSnapshot(`Updated Centroids (Max Shift: ${maxDisplacement.toFixed(2)}px)`);
    return { maxDisplacement, isConverged: this.isConverged };
  }

  /**
   * Execute single atomic algorithm step
   */
  step() {
    if (this.isConverged) return { phase: this.phase, converged: true };

    if (this.phase === 'IDLE') {
      this.initializeCentroids();
      return { phase: this.phase, converged: false };
    }

    if (this.phase === 'INITIALIZED' || this.phase === 'UPDATE') {
      this.assignPoints();
      return { phase: this.phase, converged: false };
    }

    if (this.phase === 'ASSIGNMENT') {
      const updateResult = this.updateCentroids();
      return { phase: this.phase, converged: updateResult.isConverged };
    }

    return { phase: this.phase, converged: this.isConverged };
  }

  /**
   * Run until convergence
   */
  runToConvergence() {
    if (this.phase === 'IDLE') {
      this.initializeCentroids();
    }
    while (!this.isConverged && this.iteration < this.maxIterations) {
      this.step();
    }
    return this.isConverged;
  }

  /**
   * Recompute assignments without creating new step (used when metric changes)
   */
  recomputeAssignments() {
    if (this.centroids.length === 0) return;
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      let minDistance = Infinity;
      let closest = 0;
      for (let c = 0; c < this.centroids.length; c++) {
        const dist = this.distanceFn(p, this.centroids[c]);
        if (dist < minDistance) {
          minDistance = dist;
          closest = c;
        }
      }
      this.assignments[i] = closest;
    }
  }

  /**
   * Calculate Inertia / WCSS (Within-Cluster Sum of Squares)
   */
  calculateWCSS() {
    if (this.centroids.length === 0 || this.points.length === 0) return 0;
    let wcss = 0;
    for (let i = 0; i < this.points.length; i++) {
      const clusterIdx = this.assignments[i];
      if (clusterIdx >= 0 && clusterIdx < this.centroids.length) {
        const c = this.centroids[clusterIdx];
        wcss += (this.points[i].x - c.x) ** 2 + (this.points[i].y - c.y) ** 2;
      }
    }
    return wcss;
  }

  /**
   * Save snapshot for step undo/redo
   */
  saveStateSnapshot(actionDescription = '') {
    // If we were browsing back in history and did a new step, truncate ahead
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1);
    }

    const snapshot = {
      centroids: this.centroids.map(c => ({ x: c.x, y: c.y })),
      assignments: [...this.assignments],
      trajectories: this.centroidTrajectories.map(trail => trail.map(pt => ({ x: pt.x, y: pt.y }))),
      iteration: this.iteration,
      phase: this.phase,
      isConverged: this.isConverged,
      wcss: this.calculateWCSS(),
      actionDescription
    };

    this.history.push(snapshot);
    this.historyIndex = this.history.length - 1;
  }

  stepBack() {
    if (this.historyIndex > 0) {
      this.historyIndex--;
      this.restoreFromSnapshot(this.history[this.historyIndex]);
      return true;
    }
    return false;
  }

  stepForward() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex++;
      this.restoreFromSnapshot(this.history[this.historyIndex]);
      return true;
    }
    return false;
  }

  restoreFromSnapshot(snap) {
    this.centroids = snap.centroids.map(c => ({ x: c.x, y: c.y }));
    this.assignments = [...snap.assignments];
    this.centroidTrajectories = snap.trajectories.map(trail => trail.map(pt => ({ x: pt.x, y: pt.y })));
    this.iteration = snap.iteration;
    this.phase = snap.phase;
    this.isConverged = snap.isConverged;
  }
}
