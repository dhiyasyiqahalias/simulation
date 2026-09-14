# 🎯 Interactive K-Means Clustering Simulation Lab

A web-based simulation and exploratory data lab for visualizing, experimenting with, and diagnosing the **K-Means Clustering** algorithm in real time.

---

## 🌟 Key Features

1. **High-Performance 2D Canvas**:
   - 60 FPS real-time Voronoi partition boundary rendering.
   - Smooth animated centroid trajectories with trailing motion paths.
   - Color-coded cluster assignments with optional distance connector lines.
   - Canvas interaction tools: **Brush/Spray points**, **Eraser**, **Manual initial centroid placement**, and **Hover coordinate inspector**.

2. **Algorithm Engine & Step-by-Step Playback**:
   - Full state machine: `INITIALIZATION` $\rightarrow$ `ASSIGNMENT` $\rightarrow$ `CENTROID UPDATE` $\rightarrow$ `CONVERGENCE`.
   - Time-travel debugger: Step Forward, Step Backward, Auto-Run to Convergence, Variable Playback Speed ($0.2\times - 5.0\times$).
   - Multiple distance metrics: **Euclidean ($L_2$)**, **Manhattan ($L_1$)**, and **Chebyshev ($L_\infty$)**.
   - Multiple initialization strategies: **K-Means++ ($D^2$ weighting)**, **Random Points (Forgy)**, **Random Uniform**, and **Manual Canvas Selection**.

3. **Built-in Dataset Presets**:
   - **Gaussian Blobs**: Standard multi-cluster testing.
   - **Concentric Rings & Two Moons**: Demonstrate geometric manifold limitations of partition-based clustering.
   - **Anisotropic**: Sheared elliptical clusters demonstrating distance metric assumptions.
   - **Varied Density**: Demonstrate sensitivity to disparate cluster densities.
   - **Smiley Face**: Non-convex shape clustering.
   - **Custom Data**: Spray custom point clouds or import/export CSV datasets.

4. **Quantitative Analytics Dashboard**:
   - **WCSS Inertia Curve**: Live iteration tracking of the Within-Cluster Sum of Squares.
   - **Silhouette Analysis**: Sample-level and per-cluster silhouette coefficient distribution plot $[-1, +1]$.
   - **Elbow Method Analyzer**: One-click sweep across $K=1 \dots 10$ with automatic optimal $K$ detection.
   - **Cluster Diagnostics Table**: Real-time cluster sizes, percentages, coordinates $(\mu_x, \mu_y)$, and standard deviations.

5. **Educational Guide**:
   - In-app mathematical formulation and objective function guide.

---

## 🚀 Running the Simulation Locally

You can serve this folder with any static HTTP server (e.g. Python, Node `http-server`, or Vite).

### Using Python:
```bash
python -m http.server 3000
```
Then navigate to `http://localhost:3000` in your browser.

### Keyboard Shortcuts:
- `Space`: Play / Pause
- `Right Arrow`: Step Forward
- `Left Arrow`: Step Backward
- `R`: Re-seed Centroids
- `G`: Re-generate Dataset
