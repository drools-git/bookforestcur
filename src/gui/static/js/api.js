/* Fetch wrappers for the FastAPI backend */

const API = {
  async getGalaxyGraph() {
    const r = await fetch('/api/graph/galaxy');
    if (!r.ok) throw new Error('Failed to load galaxy graph');
    return r.json();
  },

  async getTreeGraph() {
    const r = await fetch('/api/graph/tree');
    if (!r.ok) throw new Error('Failed to load tree graph');
    return r.json();
  },

  async getStats() {
    const r = await fetch('/api/stats');
    if (!r.ok) return {};
    return r.json();
  },

  async search(q) {
    if (!q || q.length < 2) return [];
    const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    if (!r.ok) return [];
    const data = await r.json();
    return data.matches || [];
  },

  async updateBookmarkCategory(bookmarkId, categoryPath) {
    const r = await fetch(`/api/bookmarks/${bookmarkId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category_path: categoryPath }),
    });
    if (!r.ok) throw new Error('Failed to update bookmark category');
    return r.json();
  },
};
