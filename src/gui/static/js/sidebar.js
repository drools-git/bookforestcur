/* Sidebar — shows node metadata on click */

const Sidebar = {
  _el: null,
  _content: null,

  init() {
    this._el = document.getElementById('sidebar');
    this._content = document.getElementById('sidebar-content');
    document.getElementById('sidebar-close').addEventListener('click', () => this.hide());
  },

  show(node) {
    this._content.innerHTML = node.type === 'bookmark'
      ? this._bookmarkHTML(node)
      : this._categoryHTML(node);
    this._el.classList.remove('hidden');
  },

  hide() {
    this._el.classList.add('hidden');
  },

  _bookmarkHTML(n) {
    const pathHtml = (n.category_path || [])
      .map(p => `<span class="sb-path-part">${p}</span>`)
      .join('');

    return `
      <div class="sb-title">${_esc(n.title || n.label)}</div>
      <div class="sb-domain-badge">${_esc(n.domain || '')}</div>
      <div class="sb-label">URL</div>
      <a class="sb-link" href="${_esc(n.url)}" target="_blank" rel="noopener">${_esc(n.url)}</a>
      <div class="sb-label">Category Path</div>
      <div class="sb-path">${pathHtml || '<span class="sb-path-part">MISC</span>'}</div>
    `;
  },

  _categoryHTML(n) {
    const confPct = Math.round((n.confidence || 1) * 100);
    const levelName = { 1: 'Root', 2: 'Domain', 3: 'Subdomain', 4: 'Topic' }[n.level] || `L${n.level}`;
    return `
      <div class="sb-title">${_esc(n.label)}</div>
      <div class="sb-domain-badge">Level ${n.level} — ${levelName}</div>
      <div class="sb-label">Bookmarks</div>
      <div class="sb-value">${n.member_count ?? '–'}</div>
      <div class="sb-label">Confidence</div>
      <div class="sb-confidence">
        <div class="conf-bar"><div class="conf-fill" style="width:${confPct}%"></div></div>
        <span>${confPct}%</span>
      </div>
      ${n.is_preset ? '<div class="sb-label" style="margin-top:12px;">📌 Preset root bucket</div>' : ''}
    `;
  },
};

function _esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
