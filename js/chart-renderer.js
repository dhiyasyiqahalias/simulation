/**
 * chart-renderer.js - High-performance canvas chart visualizers for Inertia, Silhouette & Elbow
 */

export class ChartRenderer {
  /**
   * Render Inertia / WCSS History Chart
   */
  static renderInertiaChart(canvas, history) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement.clientWidth || 300;
    const height = canvas.parentElement.clientHeight || 160;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    if (!history || history.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No history data yet', width / 2, height / 2);
      return;
    }

    const padding = { top: 20, right: 25, bottom: 25, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const values = history.map(h => h.wcss || 0);
    const maxVal = Math.max(...values, 100);
    const minVal = Math.min(...values, 0);

    // Draw Grid Lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 3; i++) {
      const y = padding.top + (chartH / 3) * i;
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);

      // Y-axis label (abbreviated e.g. 1.2M or 45k)
      const val = maxVal - (maxVal - minVal) * (i / 3);
      ctx.fillStyle = '#64748b';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const label = val >= 1000000 ? `${(val / 1000000).toFixed(1)}M` : val >= 1000 ? `${(val / 1000).toFixed(0)}k` : val.toFixed(0);
      ctx.fillText(label, padding.left - 6, y);
    }
    ctx.stroke();

    // Draw Area under curve
    const points = values.map((val, idx) => {
      const x = padding.left + (values.length === 1 ? chartW / 2 : (idx / (values.length - 1)) * chartW);
      const y = padding.top + chartH - ((val - minVal) / (maxVal - minVal || 1)) * chartH;
      return { x, y };
    });

    const grad = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    grad.addColorStop(0, 'rgba(99, 102, 241, 0.35)');
    grad.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.lineTo(points[points.length - 1].x, padding.top + chartH);
    ctx.lineTo(points[0].x, padding.top + chartH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Draw Line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Draw points & active pulse
    points.forEach((pt, idx) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, idx === points.length - 1 ? 4.5 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = idx === points.length - 1 ? '#38bdf8' : '#818cf8';
      ctx.fill();
      ctx.strokeStyle = '#090d16';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // X-axis label
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`Step ${history.length - 1}`, width - padding.right, height - padding.bottom + 6);
  }

  /**
   * Render Silhouette Plot (Horizontal bars grouped by cluster)
   */
  static renderSilhouetteChart(canvas, silhouetteData, clusterColors) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement.clientWidth || 300;
    const height = canvas.parentElement.clientHeight || 160;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (!silhouetteData || !silhouetteData.sampleScores || silhouetteData.sampleScores.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Silhouette requires K ≥ 2', width / 2, height / 2);
      return;
    }

    const padding = { top: 15, right: 15, bottom: 25, left: 35 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Zero line is at x = 0 (maps from -1 to 1)
    const toCanvasX = (score) => padding.left + ((score + 1) / 2) * chartW;
    const zeroX = toCanvasX(0);

    // Draw Grid & Guides
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
    ctx.lineWidth = 1;
    [-1, -0.5, 0, 0.5, 1].forEach(tick => {
      const x = toCanvasX(tick);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();

      ctx.fillStyle = tick === 0 ? '#94a3b8' : '#64748b';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(tick.toString(), x, height - padding.bottom + 4);
    });

    // Sort samples by cluster, then by score descending
    const sorted = [...silhouetteData.sampleScores].sort((a, b) => {
      if (a.cluster !== b.cluster) return a.cluster - b.cluster;
      return b.score - a.score;
    });

    const barHeight = Math.max(1, chartH / sorted.length);
    let currentY = padding.top;

    sorted.forEach((item, idx) => {
      const color = clusterColors[item.cluster % clusterColors.length] || '#6366f1';
      const x1 = zeroX;
      const x2 = toCanvasX(item.score);

      ctx.fillStyle = color;
      ctx.fillRect(Math.min(x1, x2), currentY, Math.abs(x2 - x1), barHeight);
      currentY += barHeight;
    });

    // Draw Average Silhouette Score Red Dashed Line
    const avgX = toCanvasX(silhouetteData.overall);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(avgX, padding.top);
    ctx.lineTo(avgX, height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Avg label
    ctx.fillStyle = '#f43f5e';
    ctx.font = 'bold 9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`Avg: ${silhouetteData.overall.toFixed(2)}`, avgX, padding.top - 5);
  }

  /**
   * Render Elbow Method Curve (Inertia vs K)
   */
  static renderElbowChart(canvas, elbowData) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement.clientWidth || 300;
    const height = canvas.parentElement.clientHeight || 160;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const { curve, optimalK } = elbowData || { curve: [], optimalK: 1 };
    if (!curve || curve.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Run Elbow Sweep" to compute', width / 2, height / 2);
      return;
    }

    const padding = { top: 20, right: 25, bottom: 25, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const maxInertia = Math.max(...curve.map(c => c.inertia));
    const minInertia = Math.min(...curve.map(c => c.inertia));

    // Draw Grid Lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 3; i++) {
      const y = padding.top + (chartH / 3) * i;
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);

      const val = maxInertia - (maxInertia - minInertia) * (i / 3);
      ctx.fillStyle = '#64748b';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const label = val >= 1000000 ? `${(val / 1000000).toFixed(1)}M` : val >= 1000 ? `${(val / 1000).toFixed(0)}k` : val.toFixed(0);
      ctx.fillText(label, padding.left - 6, y);
    }
    ctx.stroke();

    const points = curve.map((pt, idx) => {
      const x = padding.left + (idx / (curve.length - 1)) * chartW;
      const y = padding.top + chartH - ((pt.inertia - minInertia) / (maxInertia - minInertia || 1)) * chartH;
      return { x, y, k: pt.k, inertia: pt.inertia };
    });

    // Draw Line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Draw Points and optimal callout
    points.forEach(pt => {
      const isOptimal = pt.k === optimalK;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isOptimal ? 6 : 3, 0, Math.PI * 2);
      ctx.fillStyle = isOptimal ? '#10b981' : '#38bdf8';
      ctx.fill();
      ctx.strokeStyle = '#090d16';
      ctx.lineWidth = 2;
      ctx.stroke();

      if (isOptimal) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Badge
        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 10px Outfit, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`Optimal K=${pt.k}`, pt.x, pt.y - 14);
      }

      // X labels (K values)
      ctx.fillStyle = isOptimal ? '#10b981' : '#94a3b8';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`K=${pt.k}`, pt.x, height - padding.bottom + 6);
    });
  }
}
