/* App init and view router */

(async () => {
  Sidebar.init();

  const container = document.getElementById('sigma-container');
  const loading = document.getElementById('loading');
  const loadingMsg = document.getElementById('loading-msg');

  // Current active view module
  let currentView = null;
  let currentViewName = 'galaxy';
  let galaxyData = null;
  let treeData = null;

  // ── Load stats ───────────────────────────────────────────────────────
  try {
    const stats = await API.getStats();
    document.getElementById('stat-active').textContent =
      stats.active_bookmarks?.toLocaleString() ?? '–';
    document.getElementById('stat-cats').textContent =
      stats.total_categories?.toLocaleString() ?? '–';
  } catch (_) {}

  // ── Load graph data ──────────────────────────────────────────────────
  try {
    loadingMsg.textContent = 'Loading galaxy graph…';
    galaxyData = await API.getGalaxyGraph();
  } catch (e) {
    loadingMsg.textContent = 'Error loading graph. Make sure the pipeline has run first.';
    console.error(e);
    return;
  }

  // ── Mount Galaxy view ────────────────────────────────────────────────
  await switchView('galaxy');
  loading.classList.add('hidden');

  // ── View toggle ──────────────────────────────────────────────────────
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const view = btn.dataset.view;
      if (view === currentViewName) return;
      document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loading.classList.remove('hidden');
      loadingMsg.textContent = `Loading ${view} view…`;
      await switchView(view);
      loading.classList.add('hidden');
    });
  });

  async function switchView(name) {
    if (currentView) currentView.destroy();
    Sidebar.hide();

    if (name === 'galaxy') {
      currentView = GalaxyView;
      await GalaxyView.mount(container, galaxyData);
    } else if (name === 'tree') {
      if (!treeData) {
        loadingMsg.textContent = 'Computing tree layout…';
        treeData = await API.getTreeGraph();
      }
      currentView = TreeView;
      await TreeView.mount(container, treeData);
    } else if (name === 'focus') {
      currentView = FocusView;
      await FocusView.mount(container, galaxyData);
    }
    currentViewName = name;
  }

  // ── Search ───────────────────────────────────────────────────────────
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');

  // Build label lookup from galaxy data for instant local search
  const nodeLabel = {};
  (galaxyData?.nodes || []).forEach(n => {
    nodeLabel[n.key] = n.label || n.title || n.key;
  });

  let searchDebounce = null;
  searchInput.addEventListener('input', () => {
    clearTimeout(searchDebounce);
    const q = searchInput.value.trim();
    if (!q) {
      searchResults.classList.remove('open');
      currentView?.resetHighlight();
      return;
    }
    searchDebounce = setTimeout(async () => {
      const matches = await API.search(q);
      renderSearchResults(matches);
      currentView?.highlight(matches);
    }, 250);
  });

  searchInput.addEventListener('blur', () => {
    setTimeout(() => searchResults.classList.remove('open'), 200);
  });

  function renderSearchResults(keys) {
    searchResults.innerHTML = '';
    if (!keys.length) {
      searchResults.classList.remove('open');
      return;
    }
    keys.slice(0, 30).forEach(key => {
      const div = document.createElement('div');
      div.className = 'search-item';
      div.textContent = nodeLabel[key] || key;
      div.addEventListener('click', () => {
        // Find the node data and show sidebar
        const n = (galaxyData?.nodes || []).find(x => x.key === key);
        if (n) Sidebar.show(n);
        searchResults.classList.remove('open');
      });
      searchResults.appendChild(div);
    });
    searchResults.classList.add('open');
  }
})();
