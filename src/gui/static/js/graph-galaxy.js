/*
 * Galaxy View — force-directed layout using ForceAtlas2.
 * Bookmark nodes = small dots, category nodes = larger labeled circles.
 * Colors are assigned per L1 root bucket using a fixed HSL palette.
 */

const GalaxyView = {
  _renderer: null,
  _graph: null,

  // ------------------------------------------------------------------
  // Node color by root bucket name
  // ------------------------------------------------------------------
  _rootHue: {},

  _getColor(node) {
    const path = node.category_path || [];
    const root = (path[0] || node.label || '').toUpperCase();
    if (!this._rootHue[root]) {
      // Assign deterministic hue from hash
      let h = 0;
      for (let i = 0; i < root.length; i++) h = (h * 31 + root.charCodeAt(i)) % 360;
      this._rootHue[root] = h;
    }
    const hue = this._rootHue[root];
    const level = node.level || 5;
    // Categories brighter, bookmarks dimmer
    const lightness = level <= 4 ? 60 - level * 5 : 30;
    const saturation = level <= 4 ? 70 : 45;
    return `hsl(${hue},${saturation}%,${lightness}%)`;
  },

  _getSize(node) {
    if (node.type === 'bookmark') return 3;
    return Math.max(8, 32 - node.level * 6);
  },

  // ------------------------------------------------------------------
  // Build graphology graph
  // ------------------------------------------------------------------
  _buildGraph(data) {
    const g = new graphology.Graph({ multi: false, type: 'directed' });

    for (const n of data.nodes) {
      g.addNode(n.key, {
        label: n.type === 'bookmark' ? '' : n.label,  // labels only on categories
        size: this._getSize(n),
        color: this._getColor(n),
        // Store original data for sidebar
        _data: n,
        // Random initial position (ForceAtlas2 will move them)
        x: (Math.random() - 0.5) * 1000,
        y: (Math.random() - 0.5) * 1000,
      });
    }

    for (const e of data.edges) {
      if (g.hasNode(e.source) && g.hasNode(e.target)) {
        try {
          g.addEdge(e.source, e.target, {
            color: '#2a2d3a',
            size: e.type === 'hierarchy' ? 1.5 : 0.5,
          });
        } catch (_) { /* duplicate edge — skip */ }
      }
    }
    return g;
  },

  // ------------------------------------------------------------------
  // Mount
  // ------------------------------------------------------------------
  async mount(container, data) {
    this.destroy();
    this._graph = this._buildGraph(data);

    // Run ForceAtlas2 for a fixed number of iterations (off-screen)
    const totalNodes = this._graph.order;
    const iterations = Math.min(200, Math.max(50, Math.floor(5000 / totalNodes)));

    graphologyLayoutForceAtlas2.assign(this._graph, {
      iterations,
      settings: graphologyLayoutForceAtlas2.inferSettings(this._graph),
    });

    this._renderer = new Sigma(this._graph, container, {
      renderEdgeLabels: false,
      labelThreshold: 8,  // only show category labels (size >= 8)
      defaultEdgeColor: '#2a2d3a',
      labelColor: { color: '#e2e4ef' },
      labelSize: 11,
      // Show full label on hover for bookmark nodes
      nodeReducer: (node, data) => data,
      edgeReducer: (edge, data) => data,
    });

    this._bindEvents();
  },

  // ------------------------------------------------------------------
  // Events
  // ------------------------------------------------------------------
  _bindEvents() {
    const renderer = this._renderer;
    const graph = this._graph;

    // Tooltip element (shared)
    let tooltip = document.getElementById('node-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'node-tooltip';
      document.body.appendChild(tooltip);
    }

    renderer.on('enterNode', ({ node, event }) => {
      const attrs = graph.getNodeAttributes(node);
      const d = attrs._data;
      tooltip.style.display = 'block';
      tooltip.textContent = d.label || d.title || node;
      tooltip.style.left = (event.original.clientX + 12) + 'px';
      tooltip.style.top = (event.original.clientY + 12) + 'px';
    });

    renderer.on('moveBody', ({ event }) => {
      if (tooltip.style.display === 'block') {
        tooltip.style.left = (event.original.clientX + 12) + 'px';
        tooltip.style.top = (event.original.clientY + 12) + 'px';
      }
    });

    renderer.on('leaveNode', () => {
      tooltip.style.display = 'none';
    });

    renderer.on('clickNode', ({ node }) => {
      const attrs = graph.getNodeAttributes(node);
      Sidebar.show(attrs._data);
    });
  },

  // ------------------------------------------------------------------
  // Highlight matching nodes from search
  // ------------------------------------------------------------------
  highlight(matchKeys) {
    if (!this._renderer || !this._graph) return;
    const matchSet = new Set(matchKeys);
    this._renderer.setSetting('nodeReducer', (node, data) => {
      if (matchKeys.length === 0) return data;
      return matchSet.has(node)
        ? { ...data, zIndex: 1 }
        : { ...data, color: '#1e2130', label: '' };
    });
    this._renderer.setSetting('edgeReducer', (edge, data) => {
      return matchKeys.length === 0 ? data : { ...data, color: '#1e2130' };
    });
    this._renderer.refresh();
  },

  resetHighlight() {
    if (!this._renderer) return;
    this._renderer.setSetting('nodeReducer', (_, d) => d);
    this._renderer.setSetting('edgeReducer', (_, d) => d);
    this._renderer.refresh();
  },

  destroy() {
    if (this._renderer) {
      this._renderer.kill();
      this._renderer = null;
    }
    this._graph = null;
  },
};
