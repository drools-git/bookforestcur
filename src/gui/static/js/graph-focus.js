/*
 * Focus View — click any node to zoom into its branch.
 * All nodes outside the branch are faded to 10% opacity.
 * Breadcrumb bar shows the path; Back button returns.
 */

const FocusView = {
  _renderer: null,
  _graph: null,
  _history: [],   // stack of { focusKey, cameraState }

  _getColor(node) { return GalaxyView._getColor(node); },
  _getSize(node) { return GalaxyView._getSize(node); },

  _buildGraph(data) {
    // Re-use the galaxy graph structure but with better initial spread
    return GalaxyView._buildGraph.call(GalaxyView, data);
  },

  async mount(container, data) {
    this.destroy();
    this._history = [];
    this._graph = this._buildGraph(data);

    // Run ForceAtlas2 to produce a layout
    const iters = Math.min(150, Math.max(50, Math.floor(4000 / this._graph.order)));
    graphologyLayoutForceAtlas2.assign(this._graph, {
      iterations: iters,
      settings: graphologyLayoutForceAtlas2.inferSettings(this._graph),
    });

    this._renderer = new Sigma(this._graph, container, {
      renderEdgeLabels: false,
      labelThreshold: 8,
      labelColor: { color: '#e2e4ef' },
      labelSize: 11,
    });

    this._bindEvents();
    this._hideBreadcrumb();
  },

  _bindEvents() {
    const renderer = this._renderer;
    const graph = this._graph;

    let tooltip = document.getElementById('node-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'node-tooltip';
      document.body.appendChild(tooltip);
    }

    renderer.on('enterNode', ({ node, event }) => {
      tooltip.style.display = 'block';
      tooltip.textContent = graph.getNodeAttribute(node, 'label') || node;
      tooltip.style.left = (event.original.clientX + 12) + 'px';
      tooltip.style.top = (event.original.clientY + 12) + 'px';
    });

    renderer.on('leaveNode', () => { tooltip.style.display = 'none'; });

    renderer.on('clickNode', ({ node }) => {
      const attrs = graph.getNodeAttributes(node);
      const d = attrs._data;

      if (d.type === 'bookmark') {
        Sidebar.show(d);
        return;
      }

      // Focus on this category branch
      const prevCamera = renderer.getCamera().getState();
      this._history.push({ focusKey: node, cameraState: prevCamera });
      this._focusBranch(node, d.category_path || [d.label]);
    });
  },

  // ------------------------------------------------------------------
  // Focus on a branch
  // ------------------------------------------------------------------
  _focusBranch(rootKey, pathParts) {
    const graph = this._graph;

    // Collect all ancestors + descendants of rootKey
    const inBranch = new Set([rootKey]);

    // Ancestors (walk up to root)
    const ancestors = (key) => {
      graph.inNeighbors(key).forEach(n => {
        if (!inBranch.has(n)) {
          inBranch.add(n);
          ancestors(n);
        }
      });
    };

    // Descendants (walk down to leaves)
    const descendants = (key) => {
      graph.outNeighbors(key).forEach(n => {
        if (!inBranch.has(n)) {
          inBranch.add(n);
          descendants(n);
        }
      });
    };

    ancestors(rootKey);
    descendants(rootKey);

    // Fade non-branch nodes
    this._renderer.setSetting('nodeReducer', (node, data) => {
      return inBranch.has(node)
        ? data
        : { ...data, color: 'rgba(42,45,58,0.15)', label: '', size: data.size * 0.4 };
    });
    this._renderer.setSetting('edgeReducer', (edge, data) => {
      const [s, t] = this._graph.extremities(edge);
      return (inBranch.has(s) && inBranch.has(t))
        ? data
        : { ...data, color: 'rgba(42,45,58,0.1)' };
    });
    this._renderer.refresh();

    // Camera zoom to fit branch
    this._zoomToBranch([...inBranch]);

    // Update breadcrumb
    this._showBreadcrumb(pathParts);
  },

  _zoomToBranch(keys) {
    const graph = this._graph;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    keys.forEach(k => {
      const x = graph.getNodeAttribute(k, 'x');
      const y = graph.getNodeAttribute(k, 'y');
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const span = Math.max(maxX - minX, maxY - minY, 1);
    const ratio = 600 / span;  // target ~600 canvas units

    this._renderer.getCamera().animate(
      { x: cx, y: cy, ratio: Math.min(ratio, 5) },
      { duration: 400 }
    );
  },

  // ------------------------------------------------------------------
  // Breadcrumb
  // ------------------------------------------------------------------
  _showBreadcrumb(parts) {
    const el = document.getElementById('breadcrumb');
    el.classList.remove('hidden');
    el.innerHTML = '';

    parts.forEach((part, i) => {
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'breadcrumb-sep';
        sep.textContent = '›';
        el.appendChild(sep);
      }
      const span = document.createElement('span');
      span.className = 'breadcrumb-part';
      span.textContent = part;
      el.appendChild(span);
    });

    const backBtn = document.createElement('button');
    backBtn.id = 'breadcrumb-back';
    backBtn.textContent = '← Back';
    backBtn.addEventListener('click', () => this._goBack());
    el.appendChild(backBtn);
  },

  _hideBreadcrumb() {
    document.getElementById('breadcrumb').classList.add('hidden');
  },

  // ------------------------------------------------------------------
  // Back navigation
  // ------------------------------------------------------------------
  _goBack() {
    this._history.pop();
    if (this._history.length === 0) {
      // Reset to full view
      this._renderer.setSetting('nodeReducer', (_, d) => d);
      this._renderer.setSetting('edgeReducer', (_, d) => d);
      this._renderer.refresh();
      this._renderer.getCamera().animate({ x: 0, y: 0, ratio: 1 }, { duration: 300 });
      this._hideBreadcrumb();
    } else {
      const prev = this._history[this._history.length - 1];
      this._focusBranch(prev.focusKey, []);
    }
  },

  highlight(matchKeys) {
    GalaxyView.highlight.call({ _renderer: this._renderer, _graph: this._graph }, matchKeys);
  },

  resetHighlight() {
    GalaxyView.resetHighlight.call({ _renderer: this._renderer });
  },

  destroy() {
    if (this._renderer) {
      this._renderer.kill();
      this._renderer = null;
    }
    this._graph = null;
    this._history = [];
    this._hideBreadcrumb();
  },
};
