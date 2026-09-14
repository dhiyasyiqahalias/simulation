/**
 * chat-assistant.js - Interactive K-Means Q&A and Simulation Assistant
 */

export class ChatAssistant {
  constructor(appInstance) {
    this.app = appInstance;
    this.isOpen = false;
    this.messages = [
      {
        role: 'assistant',
        text: `👋 **Hi there!** I'm your interactive **K-Means AI Assistant**. You can ask me any question about the algorithm, math formulas, why clusters move, or how to interpret your live simulation results!`
      }
    ];

    this.initDOM();
    this.bindEvents();
    this.renderMessages();
  }

  initDOM() {
    this.chatToggleBtn = document.getElementById('chatToggleBtn');
    this.chatPanel = document.getElementById('chatPanel');
    this.chatCloseBtn = document.getElementById('chatCloseBtn');
    this.chatMessagesContainer = document.getElementById('chatMessagesContainer');
    this.chatForm = document.getElementById('chatForm');
    this.chatInput = document.getElementById('chatInput');
    this.chatQuickPills = document.getElementById('chatQuickPills');
  }

  bindEvents() {
    this.chatToggleBtn?.addEventListener('click', () => this.toggleChat());
    this.chatCloseBtn?.addEventListener('click', () => this.closeChat());

    this.chatForm?.addEventListener('submit', (e) => {
      e.preventDefault();
      const query = this.chatInput.value.trim();
      if (query) {
        this.handleUserMessage(query);
        this.chatInput.value = '';
      }
    });

    this.chatQuickPills?.addEventListener('click', (e) => {
      const pill = e.target.closest('.quick-pill');
      if (pill) {
        const question = pill.dataset.question || pill.textContent.trim();
        this.handleUserMessage(question);
      }
    });
  }

  toggleChat() {
    this.isOpen = !this.isOpen;
    if (this.isOpen) {
      this.chatPanel.classList.add('open');
      this.chatToggleBtn.classList.add('active');
      setTimeout(() => this.chatInput.focus(), 200);
      this.scrollToBottom();
    } else {
      this.closeChat();
    }
  }

  closeChat() {
    this.isOpen = false;
    this.chatPanel.classList.remove('open');
    this.chatToggleBtn.classList.remove('active');
  }

  handleUserMessage(text) {
    // 1. Append user message
    this.messages.push({ role: 'user', text });
    this.renderMessages();

    // 2. Show typing indicator
    this.showTypingIndicator();

    // 3. Generate response with small natural delay
    setTimeout(() => {
      this.hideTypingIndicator();
      const response = this.generateResponse(text);
      this.messages.push({ role: 'assistant', text: response });
      this.renderMessages();
      this.scrollToBottom();
    }, 450);
  }

  showTypingIndicator() {
    const existing = document.getElementById('chatTypingIndicator');
    if (existing) return;

    const typingEl = document.createElement('div');
    typingEl.id = 'chatTypingIndicator';
    typingEl.className = 'chat-message chat-message-assistant';
    typingEl.innerHTML = `
      <div class="chat-avatar">AI</div>
      <div class="chat-bubble typing-bubble">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    `;
    this.chatMessagesContainer.appendChild(typingEl);
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    const typingEl = document.getElementById('chatTypingIndicator');
    if (typingEl) typingEl.remove();
  }

  /**
   * Comprehensive Context-Aware NLP Response Generator
   */
  generateResponse(query) {
    const q = query.toLowerCase().trim();
    const engine = this.app.engine;
    const currentK = engine.k;
    const currentPoints = engine.points.length;
    const currentIter = engine.iteration;
    const currentMetric = engine.metricName;
    const currentInit = engine.initStrategy;
    const isConverged = engine.isConverged;
    const currentWCSS = Math.round(engine.calculateWCSS());
    const currentPreset = this.app.currentPreset;

    // 0. Malaysian Sustainable Tourism Specific Queries
    if (q.includes('tourism') || q.includes('sustainable') || q.includes('malaysia') || q.includes('dosm') || q.includes('state') || q.includes('putrajaya') || q.includes('melaka') || q.includes('selangor') || q.includes('johor') || q.includes('density') || q.includes('overcrowded') || q.includes('tier')) {
      return `🇲🇾 **Sustainable Tourism 2023 Analysis (DOSM Datathon):**\n\n` +
        `This dataset combines 2023 Domestic Tourism data across all **16 Malaysian States and Federal Territories** with OpenDOSM population data.\n\n` +
        `**Key Findings & Tiers:**\n` +
        `• **Overcrowded (High Density):** \`W.P. Putrajaya\` (16.0 visitors/resident), \`Melaka\` (15.1), \`Negeri Sembilan\` (12.2). These states experience heavy tourist influx relative to their local population base.\n` +
        `• **Under-visited (Low Density relative to Population):** \`Johor\` (3.8), \`Selangor\` (3.8), \`W.P. Labuan\` (3.3). Despite Selangor having the highest raw visitors (27.6M), its huge resident population (7.2M) gives it low visitor density!\n` +
        `• **Balanced (10 States):** \`W.P. KL\` (11.1), \`Pahang\` (10.0), \`Terengganu\` (9.7), \`Penang\` (7.4), \`Sarawak\` (7.2), \`Perak\` (6.7), \`Perlis\` (6.7), \`Kedah\` (6.1), \`Sabah\` (4.5), \`Kelantan\` (4.1).\n\n` +
        `💡 **Why K-Means over Simple Rules?**\n` +
        `Rule-based cutoffs use arbitrary 1.5x/0.5x thresholds. **K-Means clustering** groups states multidimensionally (e.g. Density + Spend per Visitor + Length of stay), discovering data-driven policy clusters without human bias!`;
    }

    // 1. Current Live State Queries
    if (q.includes('current') || q.includes('status') || q.includes('state') || q.includes('my simulation') || q.includes('what is happening')) {
      return `📊 **Current Simulation State:**\n` +
        `- **Clusters (K):** \`${currentK}\`\n` +
        `- **Dataset Preset:** \`${currentPreset}\` (${currentPoints} points)\n` +
        `- **Iteration:** \`${currentIter}\` (Status: ${isConverged ? '**Converged ✅**' : '**Running / In Progress ⏳**'})\n` +
        `- **Distance Metric:** \`${currentMetric}\`\n` +
        `- **Inertia (WCSS):** \`${currentWCSS.toLocaleString()}\`\n` +
        `- **Initialization:** \`${currentInit}\`\n\n` +
        (isConverged
          ? `Centroids have reached equilibrium where displacement is below tolerance $\\epsilon$.`
          : `Click **Step Forward** (or press \`Space\`) to watch the next centroid update!`);
    }

    // 2. K-Means++ Query
    if (q.includes('kmeans++') || q.includes('k-means++') || q.includes('plus plus') || q.includes('initialization')) {
      return `✨ **What is K-Means++?**\n\n` +
        `Standard K-Means picks starting centroids uniformly at random, which often places multiple centroids in the same cluster, resulting in poor local minima.\n\n` +
        `**How K-Means++ works:**\n` +
        `1. Chooses the first centroid $\\mu_1$ randomly from the dataset.\n` +
        `2. For each remaining point $x$, computes $D(x)$ (the distance to the closest already-chosen centroid).\n` +
        `3. Selects the next centroid with probability proportional to **$D(x)^2$**.\n\n` +
        `💡 **Benefit:** It guarantees an expected approximation ratio of **$O(\\log k)$** relative to the optimal clustering and converges much faster!`;
    }

    // 3. Elbow Method Query
    if (q.includes('elbow') || q.includes('choose k') || q.includes('optimal k') || q.includes('how many clusters')) {
      return `📈 **The Elbow Method for Selecting K:**\n\n` +
        `As $K$ increases, Within-Cluster Sum of Squares (**WCSS**) always decreases (dropping to 0 when $K=N$).\n\n` +
        `**The Elbow Rule:**\n` +
        `Look for the point of **maximum curvature (the "elbow")** where adding another cluster yields diminishing returns in WCSS reduction.\n\n` +
        `👉 *Tip:* Click the **Elbow Sweep** tab on the right sidebar and click **"Run Sweep"** to compute the automated curve for your current dataset!`;
    }

    // 4. Silhouette Score Query
    if (q.includes('silhouette') || q.includes('negative') || q.includes('score')) {
      return `🎯 **Understanding Silhouette Coefficients:**\n\n` +
        `The Silhouette Score measures how well-separated and cohesive clusters are, ranging from **$-1.0$ to $+1.0$**:\n\n` +
        `- **$+1.0$ (Ideal):** Point is very close to its cluster center and far from other clusters.\n` +
        `- **$0.0$ (Boundary):** Point is sitting right on the decision boundary between two clusters.\n` +
        `- **$< 0.0$ (Misassigned):** Point is on average closer to a neighboring cluster than its assigned cluster (often happens with non-spherical clusters like rings or moons)!\n\n` +
        `Check the **Silhouette** tab on the right panel to see the live per-sample breakdown!`;
    }

    // 5. Concentric Rings / Moons / Limitations
    if (q.includes('ring') || q.includes('concentric') || q.includes('moon') || q.includes('shape') || q.includes('limitation') || q.includes('fail') || q.includes('struggle')) {
      return `🔍 **Why K-Means Struggles with Concentric Rings & Moons:**\n\n` +
        `K-Means makes fundamental geometric assumptions:\n` +
        `1. **Convex / Spherical Shapes:** It assigns points purely by linear distance to a single center $\\mu_i$, which forms **Voronoi polyhedra (linear boundaries)**.\n` +
        `2. **Isotropic Variance:** It assumes clusters have equal variance in all directions.\n\n` +
        `For concentric circles, the outer ring's points are closer to the center than to other points on the opposite side of the same ring! For complex non-convex manifolds, **DBSCAN** or **Spectral Clustering** are better suited.`;
    }

    // 6. Distance Metrics (Euclidean vs Manhattan vs Chebyshev)
    if (q.includes('distance') || q.includes('metric') || q.includes('manhattan') || q.includes('euclidean') || q.includes('chebyshev')) {
      return `📏 **Distance Metrics Comparison:**\n\n` +
        `- **Euclidean ($L_2$):** $\\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$ — Straight-line standard geometric distance. Creates smooth circular Voronoi cells.\n` +
        `- **Manhattan ($L_1$):** $|x_1-x_2| + |y_1-y_2|$ — Grid/taxicab distance. Creates diamond-like decision boundaries.\n` +
        `- **Chebyshev ($L_\\infty$):** $\\max(|x_1-x_2|, |y_1-y_2|)$ — Chessboard distance. Creates rectangular decision boundaries.\n\n` +
        `You can switch distance metrics in the **Hyperparameters** dropdown on the left to see the Voronoi zones shift in real time!`;
    }

    // 7. WCSS / Inertia Formula
    if (q.includes('wcss') || q.includes('inertia') || q.includes('formula') || q.includes('objective') || q.includes('loss')) {
      return `📐 **WCSS (Within-Cluster Sum of Squares) Objective:**\n\n` +
        `$$\\mathcal{J} = \\sum_{i=1}^{K} \\sum_{x \\in C_i} \\| x - \\mu_i \\|^2$$\n\n` +
        `Where:\n` +
        `- $K$ = Number of clusters\n` +
        `- $C_i$ = Set of points assigned to cluster $i$\n` +
        `- $\\mu_i$ = Center of mass (mean) of cluster $i$\n\n` +
        `At each iteration, K-Means is guaranteed to monotonically reduce (or maintain) this objective function until reaching a local minimum!`;
    }

    // 8. Why do centroids move?
    if (q.includes('move') || q.includes('centroid') || q.includes('update') || q.includes('step')) {
      return `🔄 **How Centroids Update Each Step:**\n\n` +
        `1. **Assignment:** Each point $x$ is tagged with its closest centroid $\\mu_i$.\n` +
        `2. **Center of Mass:** The new position of centroid $\\mu_i$ is calculated as the arithmetic average of all points in its cluster:\n` +
        `$$\\mu_i^{new} = \\frac{1}{|C_i|} \\sum_{x \\in C_i} x$$\n` +
        `3. Watch the **dashed trajectory trails** on the canvas to see the exact convergence path taken!`;
    }

    // 9. Voronoi boundaries
    if (q.includes('voronoi') || q.includes('partition') || q.includes('boundary') || q.includes('region')) {
      return `🗺️ **What are Voronoi Partition Zones?**\n\n` +
        `A Voronoi diagram partitions the 2D plane into $K$ convex polygonal regions. Every coordinate inside region $i$ is closer to centroid $\\mu_i$ than to any other centroid.\n\n` +
        `In our simulation, you can toggle **Voronoi Partition Zones** in the left sidebar to visualize the active decision boundary territories!`;
    }

    // 10. Default Helpful Fallback
    return `🤖 **Great question!** K-Means is an unsupervised clustering algorithm that minimizes intra-cluster variance (WCSS) through iterative Assignment and Update steps.\n\n` +
      `Here are some things you can ask me about:\n` +
      `- *"What does the Sustainable Tourism dataset show?"*\n` +
      `- *"Why is Putrajaya/Melaka classified as overcrowded?"*\n` +
      `- *"What is K-Means++ and why use it?"*\n` +
      `- *"How does the Elbow method work?"*\n` +
      `- *"What is the current simulation state?"*`;
  }

  renderMessages() {
    if (!this.chatMessagesContainer) return;
    this.chatMessagesContainer.innerHTML = this.messages.map(msg => {
      const isUser = msg.role === 'user';
      const formattedText = this.formatMarkdown(msg.text);
      return `
        <div class="chat-message ${isUser ? 'chat-message-user' : 'chat-message-assistant'}">
          <div class="chat-avatar">${isUser ? 'You' : 'AI'}</div>
          <div class="chat-bubble">
            ${formattedText}
          </div>
        </div>
      `;
    }).join('');
  }

  formatMarkdown(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');
  }

  scrollToBottom() {
    if (this.chatMessagesContainer) {
      this.chatMessagesContainer.scrollTop = this.chatMessagesContainer.scrollHeight;
    }
  }
}
