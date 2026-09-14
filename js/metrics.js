/**
 * metrics.js - Quantitative analytics, Silhouette scores, and Elbow method calculations
 */

import { KMeansEngine, DistanceMetrics } from './kmeans.js';

export const MetricsCalculator = {
  /**
   * Calculate Silhouette Coefficients for all points and per-cluster
   */
  calculateSilhouette(points, assignments, k) {
    if (k < 2 || points.length === 0) {
      return { overall: 0, perCluster: [], sampleScores: [] };
    }

    const clusterPoints = Array.from({ length: k }, () => []);
    for (let i = 0; i < points.length; i++) {
      const c = assignments[i];
      if (c >= 0 && c < k) {
        clusterPoints[c].push({ pt: points[i], originalIndex: i });
      }
    }

    // Subsample if dataset is very large to maintain 60 FPS responsiveness
    const maxSample = 400;
    const isSampled = points.length > maxSample;
    const stride = isSampled ? Math.ceil(points.length / maxSample) : 1;

    const sampleScores = [];
    const clusterScoreSums = new Float64Array(k);
    const clusterScoreCounts = new Uint32Array(k);
    let totalScoreSum = 0;
    let totalScoreCount = 0;

    for (let i = 0; i < points.length; i += stride) {
      const cIdx = assignments[i];
      if (cIdx < 0 || cIdx >= k) continue;

      const myCluster = clusterPoints[cIdx];
      if (myCluster.length <= 1) {
        sampleScores.push({ index: i, cluster: cIdx, score: 0 });
        continue;
      }

      // 1. Calculate a(i) - mean intra-cluster distance
      let aDistSum = 0;
      for (let j = 0; j < myCluster.length; j++) {
        aDistSum += Math.hypot(points[i].x - myCluster[j].pt.x, points[i].y - myCluster[j].pt.y);
      }
      const a = aDistSum / (myCluster.length - 1);

      // 2. Calculate b(i) - min mean distance to other clusters
      let minOtherAvgDist = Infinity;
      for (let otherC = 0; otherC < k; otherC++) {
        if (otherC === cIdx) continue;
        const otherCluster = clusterPoints[otherC];
        if (otherCluster.length === 0) continue;

        let otherDistSum = 0;
        for (let j = 0; j < otherCluster.length; j++) {
          otherDistSum += Math.hypot(points[i].x - otherCluster[j].pt.x, points[i].y - otherCluster[j].pt.y);
        }
        const avgDist = otherDistSum / otherCluster.length;
        if (avgDist < minOtherAvgDist) {
          minOtherAvgDist = avgDist;
        }
      }

      const b = minOtherAvgDist === Infinity ? 0 : minOtherAvgDist;
      const denom = Math.max(a, b);
      const s = denom > 0 ? (b - a) / denom : 0;

      sampleScores.push({ index: i, cluster: cIdx, score: s });
      clusterScoreSums[cIdx] += s;
      clusterScoreCounts[cIdx]++;
      totalScoreSum += s;
      totalScoreCount++;
    }

    const perCluster = [];
    for (let c = 0; c < k; c++) {
      const avg = clusterScoreCounts[c] > 0 ? clusterScoreSums[c] / clusterScoreCounts[c] : 0;
      perCluster.push({
        cluster: c,
        score: avg,
        count: clusterPoints[c].length
      });
    }

    const overall = totalScoreCount > 0 ? totalScoreSum / totalScoreCount : 0;

    return {
      overall,
      perCluster,
      sampleScores
    };
  },

  /**
   * Run Elbow Method analysis across K = 1..10
   */
  computeElbowCurve(points, maxK = 8, runsPerK = 5, bounds = { width: 800, height: 600 }) {
    if (points.length === 0) return { curve: [], optimalK: 1 };

    const curve = [];
    const limitK = Math.min(maxK, Math.min(10, points.length));

    for (let k = 1; k <= limitK; k++) {
      let bestInertia = Infinity;

      for (let run = 0; run < runsPerK; run++) {
        const engine = new KMeansEngine({ k, metric: 'euclidean', initStrategy: 'kmeans++' });
        engine.setPoints(points);
        engine.initializeCentroids(null, bounds);
        engine.runToConvergence();
        const inertia = engine.calculateWCSS();
        if (inertia < bestInertia) {
          bestInertia = inertia;
        }
      }

      curve.push({ k, inertia: bestInertia });
    }

    // Determine optimal K using maximum distance from chord line connecting K=1 to K=limitK
    let optimalK = 1;
    if (curve.length >= 3) {
      const p1 = curve[0];
      const p2 = curve[curve.length - 1];

      let maxDist = -1;
      for (let i = 1; i < curve.length - 1; i++) {
        const p = curve[i];
        // Line equation from (p1.k, p1.inertia) to (p2.k, p2.inertia)
        // Normalized coordinates
        const normK1 = 0, normI1 = 1;
        const normK2 = 1, normI2 = 0;
        const normK = (p.k - p1.k) / (p2.k - p1.k);
        const normI = (p.inertia - p2.inertia) / (p1.inertia - p2.inertia || 1);

        // Distance from point (normK, normI) to chord line connecting (0,1) and (1,0)
        // Line: normI + normK - 1 = 0 -> distance is |normI + normK - 1| / sqrt(2)
        const chordExpectedY = 1 - normK;
        const dist = Math.abs(normI - chordExpectedY);

        if (dist > maxDist) {
          maxDist = dist;
          optimalK = p.k;
        }
      }
    }

    return { curve, optimalK };
  },

  /**
   * Cluster detailed diagnostics (sizes, variances, coordinates)
   */
  getDiagnostics(points, centroids, assignments) {
    const k = centroids.length;
    const stats = [];

    for (let c = 0; c < k; c++) {
      const clusterPts = [];
      for (let i = 0; i < points.length; i++) {
        if (assignments[i] === c) {
          clusterPts.push(points[i]);
        }
      }

      let variance = 0;
      if (clusterPts.length > 0 && centroids[c]) {
        let sumSqDist = 0;
        for (let pt of clusterPts) {
          sumSqDist += (pt.x - centroids[c].x) ** 2 + (pt.y - centroids[c].y) ** 2;
        }
        variance = sumSqDist / clusterPts.length;
      }

      stats.push({
        cluster: c,
        centroid: centroids[c] ? { x: Math.round(centroids[c].x), y: Math.round(centroids[c].y) } : { x: 0, y: 0 },
        count: clusterPts.length,
        percentage: points.length > 0 ? ((clusterPts.length / points.length) * 100).toFixed(1) : 0,
        stdDev: Math.round(Math.sqrt(variance))
      });
    }

    return stats;
  }
};
