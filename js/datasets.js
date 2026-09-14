/**
 * datasets.js - Dataset generators and point management for K-Means Simulation
 */

// Random normal (Gaussian) number generator using Box-Muller transform
function randomGaussian(mean = 0, stdev = 1) {
  let u = 1 - Math.random();
  let v = Math.random();
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + z * stdev;
}

export const DatasetGenerators = {
  /**
   * Standard Gaussian Blobs
   */
  blobs: (numPoints = 300, numCenters = 4, width = 800, height = 600, noise = 35) => {
    const points = [];
    const centers = [];
    const margin = 100;
    
    // Generate cluster centers well spaced apart
    for (let i = 0; i < numCenters; i++) {
      centers.push({
        x: margin + Math.random() * (width - 2 * margin),
        y: margin + Math.random() * (height - 2 * margin)
      });
    }

    const pointsPerCenter = Math.floor(numPoints / numCenters);
    for (let i = 0; i < numCenters; i++) {
      const count = (i === numCenters - 1) ? numPoints - (numCenters - 1) * pointsPerCenter : pointsPerCenter;
      for (let j = 0; j < count; j++) {
        const x = randomGaussian(centers[i].x, noise);
        const y = randomGaussian(centers[i].y, noise);
        points.push({
          x: Math.max(10, Math.min(width - 10, x)),
          y: Math.max(10, Math.min(height - 10, y)),
          trueCluster: i
        });
      }
    }
    return points;
  },

  /**
   * Concentric Rings / Circles (Demonstrates geometric limitation of K-Means)
   */
  concentricRings: (numPoints = 350, numRings = 3, width = 800, height = 600) => {
    const points = [];
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) * 0.42;
    const radii = [maxRadius * 0.25, maxRadius * 0.6, maxRadius];

    const ptsPerRing = Math.floor(numPoints / radii.length);

    radii.forEach((r, ringIdx) => {
      const count = (ringIdx === radii.length - 1) ? numPoints - (radii.length - 1) * ptsPerRing : ptsPerRing;
      for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radiusWithNoise = r + randomGaussian(0, 8);
        points.push({
          x: centerX + radiusWithNoise * Math.cos(angle),
          y: centerY + radiusWithNoise * Math.sin(angle),
          trueCluster: ringIdx
        });
      }
    });
    return points;
  },

  /**
   * Anisotropic / Elongated Clusters (Sheared covariance)
   */
  anisotropic: (numPoints = 320, width = 800, height = 600) => {
    const points = [];
    const centers = [
      { x: width * 0.28, y: height * 0.35, angle: 0.6, sx: 85, sy: 18 },
      { x: width * 0.72, y: height * 0.40, angle: -0.5, sx: 90, sy: 20 },
      { x: width * 0.50, y: height * 0.75, angle: 1.1, sx: 80, sy: 16 }
    ];

    const ptsPerCenter = Math.floor(numPoints / centers.length);
    centers.forEach((c, idx) => {
      const count = (idx === centers.length - 1) ? numPoints - (centers.length - 1) * ptsPerCenter : ptsPerCenter;
      const cosA = Math.cos(c.angle);
      const sinA = Math.sin(c.angle);

      for (let i = 0; i < count; i++) {
        const rawX = randomGaussian(0, c.sx);
        const rawY = randomGaussian(0, c.sy);
        const rotX = rawX * cosA - rawY * sinA;
        const rotY = rawX * sinA + rawY * cosA;

        points.push({
          x: Math.max(15, Math.min(width - 15, c.x + rotX)),
          y: Math.max(15, Math.min(height - 15, c.y + rotY)),
          trueCluster: idx
        });
      }
    });
    return points;
  },

  /**
   * Two Interlocking Moons
   */
  moons: (numPoints = 300, width = 800, height = 600) => {
    const points = [];
    const n = Math.floor(numPoints / 2);
    const radius = Math.min(width, height) * 0.28;
    const cx1 = width * 0.42;
    const cy1 = height * 0.45;
    const cx2 = width * 0.58;
    const cy2 = height * 0.58;

    // Top moon
    for (let i = 0; i < n; i++) {
      const angle = Math.PI * (i / n);
      const x = cx1 + radius * Math.cos(angle) + randomGaussian(0, 10);
      const y = cy1 - radius * Math.sin(angle) + randomGaussian(0, 10);
      points.push({ x, y, trueCluster: 0 });
    }

    // Bottom moon
    for (let i = 0; i < numPoints - n; i++) {
      const angle = Math.PI * (i / (numPoints - n));
      const x = cx2 - radius * Math.cos(angle) + randomGaussian(0, 10);
      const y = cy2 + radius * Math.sin(angle) + randomGaussian(0, 10);
      points.push({ x, y, trueCluster: 1 });
    }
    return points;
  },

  /**
   * Varied Density Blobs (Demonstrates sensitivity to cluster density)
   */
  variedDensity: (numPoints = 350, width = 800, height = 600) => {
    const points = [];
    const configs = [
      { x: width * 0.25, y: height * 0.4, count: Math.floor(numPoints * 0.55), noise: 22 }, // dense
      { x: width * 0.72, y: height * 0.35, count: Math.floor(numPoints * 0.30), noise: 55 }, // diffuse
      { x: width * 0.55, y: height * 0.75, count: Math.floor(numPoints * 0.15), noise: 75 }  // very sparse
    ];

    configs.forEach((cfg, idx) => {
      for (let i = 0; i < cfg.count; i++) {
        points.push({
          x: Math.max(15, Math.min(width - 15, randomGaussian(cfg.x, cfg.noise))),
          y: Math.max(15, Math.min(height - 15, randomGaussian(cfg.y, cfg.noise))),
          trueCluster: idx
        });
      }
    });
    return points;
  },

  /**
   * Uniform Random Noise
   */
  uniform: (numPoints = 300, width = 800, height = 600) => {
    const points = [];
    const margin = 30;
    for (let i = 0; i < numPoints; i++) {
      points.push({
        x: margin + Math.random() * (width - 2 * margin),
        y: margin + Math.random() * (height - 2 * margin),
        trueCluster: -1
      });
    }
    return points;
  },

  /**
   * Smiley Face Pattern
   */
  smiley: (numPoints = 320, width = 800, height = 600) => {
    const points = [];
    const cx = width / 2;
    const cy = height / 2;
    const faceR = Math.min(width, height) * 0.38;

    // Face outline (circle)
    const outlineCount = Math.floor(numPoints * 0.45);
    for (let i = 0; i < outlineCount; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = faceR + randomGaussian(0, 6);
      points.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), trueCluster: 0 });
    }

    // Left eye
    const eye1Count = Math.floor(numPoints * 0.12);
    for (let i = 0; i < eye1Count; i++) {
      points.push({
        x: randomGaussian(cx - faceR * 0.38, 10),
        y: randomGaussian(cy - faceR * 0.32, 10),
        trueCluster: 1
      });
    }

    // Right eye
    const eye2Count = Math.floor(numPoints * 0.12);
    for (let i = 0; i < eye2Count; i++) {
      points.push({
        x: randomGaussian(cx + faceR * 0.38, 10),
        y: randomGaussian(cy - faceR * 0.32, 10),
        trueCluster: 2
      });
    }

    // Smile curve
    const smileCount = numPoints - outlineCount - eye1Count - eye2Count;
    for (let i = 0; i < smileCount; i++) {
      const a = Math.PI * 0.15 + (Math.random() * Math.PI * 0.7);
      const r = faceR * 0.55 + randomGaussian(0, 6);
      points.push({
        x: cx + r * Math.cos(a),
        y: cy + r * Math.sin(a) + faceR * 0.1,
        trueCluster: 3
      });
    }
    return points;
  }
};
