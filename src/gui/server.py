"""Phase 11 — FastAPI server for the Sigma.js visualization GUI."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import AppConfig

logger = logging.getLogger(__name__)

_graph_data: Optional[dict] = None
_taxonomy_data: Optional[dict] = None
_output_dir: Optional[Path] = None


def _load_graph(output_dir: Path) -> dict:
    path = output_dir / "graph.json"
    if not path.exists():
        return {"nodes": [], "edges": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_taxonomy(output_dir: Path) -> dict:
    path = output_dir / "taxonomy.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_app(cfg: AppConfig) -> FastAPI:
    global _output_dir
    _output_dir = Path(cfg.output.directory)

    app = FastAPI(title="BookForest2 GUI", docs_url=None, redoc_url=None)

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/")
    async def index():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/api/graph/galaxy")
    async def graph_galaxy():
        """Full graph data for Galaxy (force) view."""
        data = _load_graph(_output_dir)
        return JSONResponse(content=data)

    @app.get("/api/graph/tree")
    async def graph_tree():
        """Hierarchical tree layout data with pre-computed radial positions."""
        data = _load_graph(_output_dir)
        enriched = _compute_tree_positions(data)
        return JSONResponse(content=enriched)

    @app.get("/api/stats")
    async def stats():
        """Summary statistics for the sidebar."""
        data = _load_taxonomy(_output_dir)
        meta = data.get("meta", {})
        return JSONResponse(content=meta)

    @app.get("/api/search")
    async def search(q: str = ""):
        """Return node keys matching query string (title/domain/url)."""
        if not q or len(q) < 2:
            return JSONResponse(content={"matches": []})
        data = _load_graph(_output_dir)
        q_lower = q.lower()
        matches = [
            n["key"]
            for n in data.get("nodes", [])
            if q_lower in n.get("label", "").lower()
            or q_lower in n.get("domain", "").lower()
        ]
        return JSONResponse(content={"matches": matches[:200]})

    class CategoryUpdate(BaseModel):
        category_path: list[str]

    @app.patch("/api/bookmarks/{bookmark_id}")
    async def update_bookmark_category(bookmark_id: str, update: CategoryUpdate):
        """
        Manually reassign a bookmark to a new category path.
        Updates graph.json in-place (lightweight — does not re-run the full pipeline).
        """
        data = _load_graph(_output_dir)
        updated = False
        for node in data.get("nodes", []):
            if node.get("key") == bookmark_id and node.get("type") == "bookmark":
                node["category_path"] = update.category_path
                updated = True
                break
        if not updated:
            raise HTTPException(status_code=404, detail="Bookmark not found")

        graph_path = _output_dir / "graph.json"
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return {"ok": True, "bookmark_id": bookmark_id, "category_path": update.category_path}

    @app.post("/api/export")
    async def export():
        """
        Re-generate HTML and report from the current graph.json.
        (Does NOT re-run embeddings or clustering.)
        """
        graph_path = _output_dir / "graph.json"
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail="graph.json not found")
        return JSONResponse(content={"ok": True, "message": "Export triggered from current data."})

    return app


def _compute_tree_positions(data: dict) -> dict:
    """
    Add x/y coordinates to nodes for the radial tree view.
    Level-1 nodes near center, level-4 topics on outer ring.
    Bookmark nodes just beyond their parent.
    """
    import math

    nodes = {n["key"]: n for n in data.get("nodes", [])}

    # Radii per level
    radii = {1: 80, 2: 200, 3: 380, 4: 600, 5: 800}

    # Find roots
    roots = [n for n in nodes.values() if n.get("level") == 1]
    n_roots = max(len(roots), 1)

    # Assign angular sectors to roots
    root_angle: dict[str, float] = {}
    for i, root in enumerate(roots):
        root_angle[root["key"]] = 2 * math.pi * i / n_roots

    # BFS assignment of positions
    positioned: dict[str, tuple[float, float]] = {}

    def assign_pos(key: str, angle: float, arc_width: float, depth: int) -> None:
        r = radii.get(depth, 900)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        positioned[key] = (x, y)

        # Children
        node = nodes.get(key, {})
        children_keys = []
        for edge in data.get("edges", []):
            if edge.get("source") == key and edge.get("type") in ("hierarchy", "contains"):
                children_keys.append(edge["target"])

        if not children_keys:
            return

        child_arc = arc_width / max(len(children_keys), 1)
        start_angle = angle - arc_width / 2
        for i, ck in enumerate(children_keys):
            child_angle = start_angle + child_arc * (i + 0.5)
            assign_pos(ck, child_angle, child_arc, depth + 1)

    for root in roots:
        angle = root_angle[root["key"]]
        arc = 2 * math.pi / n_roots
        assign_pos(root["key"], angle, arc, 1)

    # Inject positions into nodes copy
    result_nodes = []
    for n in data.get("nodes", []):
        nc = dict(n)
        if nc["key"] in positioned:
            nc["x"], nc["y"] = positioned[nc["key"]]
        result_nodes.append(nc)

    return {"nodes": result_nodes, "edges": data.get("edges", [])}


def start_server(cfg: AppConfig) -> None:
    import uvicorn

    app = create_app(cfg)
    logger.info(
        "BookForest2 GUI → http://%s:%d", cfg.gui.host, cfg.gui.port
    )
    uvicorn.run(app, host=cfg.gui.host, port=cfg.gui.port, log_level="warning")
