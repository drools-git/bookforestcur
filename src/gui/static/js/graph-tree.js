/*
 * Tree View — radial hierarchical layout.
 * Positions are pre-computed server-side (GET /api/graph/tree).
 * Bookmark leaf nodes are hidden by default; revealed on category click.
 */

const TreeView = {
  _renderer: null,
  _graph: null,
  _collapsedNodes: new Set(),

  _getColor(node) {
    return GalaxyView._getColor(node);
  },

  _getSize(node) {
    return GalaxyView._getSize(node);
  },

  _buildGraph(data) {
    const g = new graphology.Graph({ multi: false, type: 'directed' });

    for (const n of data.nodes) {
      const isBookmark = n.type === 'bookmark';
      g.addNode(n.key, {
        label: n.label,
        size: this._getSize(n),
        color: this._getColor(n),
        x: n.x ?? 0,
        y: n.y ?? 0,
        hidden: isBookmark,   // bookmarks hidden by default
        _data: n,
      });
    }

    for (const e of data.edges) {
      if (g.hasNode(e.source) && g.hasNode(e.target)) {
        try {
          g.addEdge(e.source, e.target, {
            color: e.type === 'hierarchy' ? '#3a3d50' : '#2a2d3a',
            size: e.type === 'hierarchy' ? 1.2 : 0.4,
            _type: e.type,
          });
        } catch (_) { /* skip */ }
      }
    }
    return g;
  },

  async mount(container, data) {
    this.destroy();
    this._collapsedNodes.clear();
    this._graph = this._buildGraph(data);

    this._renderer = new Sigma(this._graph, container, {
      renderEdgeLabels: false,
      labelThreshold: 6,
      labelColor: { color: '#e2e4ef' },
      labelSize: 11,
    });

    this._bindEvents();
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
      const attrs = graph.getNodeAttributes(node);
      tooltip.style.display = 'block';
      tooltip.textContent = attrs.label || node;
      tooltip.style.left = (event.original.clientX + 12) + 'px';
      tooltip.style.top = (event.original.clientY + 12) + 'px';
    });

    renderer.on('leaveNode', () => { tooltip.style.display = 'none'; });

    renderer.on('clickNode', ({ node }) => {
      const attrs = graph.getNodeAttributes(node);
      const d = attrs._data;

      if (d.type === 'category') {
        this._toggleChildren(node);
      } else {
        Sidebar.show(d);
      }
    });
  },

  _toggleChildren(categoryKey) {
    const graph = this._graph;
    const isCollapsed = this._collapsedNodes.has(categoryKey);

    // Find all descendant nodes via BFS
    const descendants = new Set();
    const queue = [categoryKey];
    while (queue.length) {
      const cur = queue.shift();
      graph.outNeighbors(cur).forEach(n => {
        if (!descendants.has(n)) {
          descendants.add(n);
          queue.push(n);
        }
      });
    }

    descendants.forEach(key => {
      graph.setNodeAttribute(key, 'hidden', !isCollapsed);
    });

    if (isCollapsed) {
      this._collapsedNodes.delete(categoryKey);
    } else {
      this._collapsedNodes.add(categoryKey);
    }

    this._renderer.refresh();
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
  },
};
