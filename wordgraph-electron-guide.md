# WordGraph + Electron: Integration Guide

This guide shows how to expose your Python `WordGraph` as an HTTP API (FastAPI) and call it from an Electron frontend, both in development and as a packaged desktop app.

-   Backend code lives under `backend/` and uses FastAPI + Uvicorn.
-   Electron app lives under `electron/` and spawns/targets the backend.

## 1) Prerequisites

-   Python 3.10+ (virtualenv recommended)
-   Node.js 18+
-   Your WordGraph code at `backend/Graphs/WordGraph/wordGraph.py`
-   `textUtils.py` at repo root (imported by WordGraph)

Install Python deps (add others you require, e.g. `sentence_transformers`).

```bash
# from repo root
python -m venv dreaming-hawk-venv
source dreaming-hawk-venv/bin/activate
pip install fastapi uvicorn[standard]
# plus your ML deps
# pip install sentence_transformers numpy networkx
```

## 2) Backend: expose a singleton WordGraph

Create a FastAPI app that holds a process‑local instance of `WordGraph` and maps class methods to HTTP endpoints.

File: `backend/app.py` (example skeleton)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Import your modules
from dreaming_hawk import wordGraph  # resolves to backend/Graphs/WordGraph/wordGraph.py via the shim

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create one process-wide WordGraph. Configure parameters as you need.
WG: Optional[wordGraph.WordGraph] = wordGraph.WordGraph(text_window_size=3)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/graph/reset")
def reset_graph():
    global WG
    WG = wordGraph.WordGraph(text_window_size=3)
    return {"ok": True}

@app.post("/api/graph/add_text")
def add_text(text: str):
    WG.add_text(text)
    return {"ok": True, "nodes": len(WG.nodes()), "edges": len(WG.edges())}

@app.get("/api/graph/window")
def get_window():
    return {"window": WG.get_window()}

@app.get("/api/graph/in_out/{word}")
def in_out(word: str):
    result = WG.in_out_edges(word)
    return result

@app.post("/api/graph/minus_node/{word}")
def minus_node(word: str):
    WG.minus_word_node(word)
    return {"ok": True}

@app.post("/api/graph/add_temporal")
def add_temporal_edge(src: str, dst: str):
    WG.add_temporal_edge(src, dst)
    return {"ok": True, "has_edge": WG.has_edge(src, dst)}

@app.post("/api/graph/tick")
def tick():
    t_before = WG.time
    WG.tick()
    return {"ok": True, "time": WG.time, "advanced": WG.time - t_before}
```

Run in dev:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 5179
```

Test quickly:

```bash
curl http://127.0.0.1:5179/health
curl -X POST "http://127.0.0.1:5179/api/graph/add_text" -H 'content-type: application/json' -d '"Hello world."'
curl http://127.0.0.1:5179/api/graph/window
```

## 3) Electron app: call the API

Create an Electron project that either spawns the packaged Python backend (in production) or points to your dev server (in development).

```bash
mkdir -p electron && cd electron
npm init -y
npm i electron electron-builder axios wait-on get-port concurrently cross-env
```

`electron/package.json` (relevant parts):

```json
{
    "name": "dreaming-hawk",
    "main": "main.js",
    "scripts": {
        "dev": "cross-env NODE_ENV=development electron .",
        "dev:all": "concurrently -k \"uvicorn backend.app:app --reload --port 5179\" \"wait-on http://127.0.0.1:5179/health && electron .\"",
        "build": "electron-builder"
    },
    "build": {
        "appId": "com.example.dreaminghawk",
        "files": ["**/*"],
        "extraResources": [
            { "from": "../dist-backend/", "to": "backend", "filter": ["**/*"] }
        ]
    },
    "dependencies": {
        "axios": "^1.7.2",
        "electron": "^30.0.0",
        "electron-builder": "^24.13.3",
        "get-port": "^7.0.0"
    },
    "devDependencies": {
        "concurrently": "^9.0.0",
        "cross-env": "^7.0.3",
        "wait-on": "^7.2.0"
    }
}
```

`electron/main.js` (minimal):

```js
const { app, BrowserWindow } = require("electron");
const path = require("path");
const axios = require("axios");

async function waitForHealth(url, timeoutMs = 10000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        try {
            if ((await axios.get(url + "/health")).status === 200) return;
        } catch {}
        await new Promise((r) => setTimeout(r, 250));
    }
    throw new Error("Backend not ready");
}

async function createWindow() {
    const isDev = process.env.NODE_ENV !== "production";
    const backendBase = isDev
        ? "http://127.0.0.1:5179"
        : "http://127.0.0.1:5179"; // updated when bundling
    await waitForHealth(backendBase);

    const win = new BrowserWindow({
        width: 1100,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, "preload.js"),
        },
    });

    await win.loadFile(path.join(__dirname, "renderer", "index.html"));
    win.webContents.executeJavaScript(
        `window.__BACKEND_URL__ = ${JSON.stringify(backendBase)};`
    );
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
});
```

`electron/renderer/index.html` (example UI):

```html
<!DOCTYPE html>
<html>
    <body>
        <h3>WordGraph Demo</h3>
        <textarea
            id="txt"
            rows="4"
            cols="60"
            placeholder="Type text..."
        ></textarea>
        <button id="btnAdd">Add Text</button>
        <pre id="out"></pre>
        <script>
            const base = window.__BACKEND_URL__ || "http://127.0.0.1:5179";
            async function refresh() {
                const w = await fetch(base + "/api/graph/window").then((r) =>
                    r.json()
                );
                document.getElementById("out").textContent = JSON.stringify(
                    w,
                    null,
                    2
                );
            }
            document.getElementById("btnAdd").onclick = async () => {
                const text = document.getElementById("txt").value;
                await fetch(base + "/api/graph/add_text", {
                    method: "POST",
                    headers: { "content-type": "application/json" },
                    body: JSON.stringify(text),
                });
                refresh();
            };
            refresh();
        </script>
    </body>
</html>
```

Dev run options:

-   Backend + Electron together:

```bash
# from repo root
cd electron
npm run dev:all
```

-   Or run backend and Electron separately:

```bash
uvicorn backend.app:app --reload --port 5179
cd electron && npm run dev
```

## 4) Mapping WordGraph methods to endpoints

Suggested minimal coverage (all implemented in `backend/app.py`):

-   `POST /api/graph/add_text` → `WordGraph.add_text(text)`
-   `GET /api/graph/window` → `WordGraph.get_window()`
-   `GET /api/graph/in_out/{word}` → `WordGraph.in_out_edges(word)`
-   `POST /api/graph/minus_node/{word}` → `WordGraph.minus_word_node(word)`
-   `POST /api/graph/add_temporal?src=...&dst=...` → `WordGraph.add_temporal_edge(src, dst)`
-   `POST /api/graph/tick` → `WordGraph.tick()` (returns time)
-   `POST /api/graph/reset` → create a new `WordGraph`

Tip: Keep response bodies small and JSON‑serializable. If you need to return the full graph, serialize nodes/edges explicitly.

## 5) Packaging for desktop

To distribute your Electron app with a Python backend, package Python into a single binary and ship it with `electron-builder`.

1. Build the backend binary with PyInstaller:

```bash
pip install pyinstaller
# Create a tiny runner that accepts --port
cat > backend/run.py <<'PY'
import argparse, uvicorn
from backend.app import app
p = argparse.ArgumentParser(); p.add_argument('--port', type=int, default=5179)
args = p.parse_args()
uvicorn.run(app, host='127.0.0.1', port=args.port)
PY

pyinstaller --onefile --name backend backend/run.py
# Copy artifact to dist-backend/ so electron-builder can include it
mkdir -p dist-backend
cp dist/run/backend dist-backend/backend  # adjust path by OS
```

2. Update `electron/main.js` to spawn the binary in production (if you prefer managed startup). Alternatively, run the backend as a separate service.

3. Build the Electron app:

```bash
cd electron
npm run build
# Artifacts under dist/
```

Signing/notarization may be needed on macOS for distribution.

## 6) Security and performance notes

-   Restrict CORS in production to loopback or your app’s file protocol.
-   Keep `nodeIntegration: false` and `contextIsolation: true` in `BrowserWindow`.
-   Use a single `WordGraph` instance unless you need per‑document isolation.
-   Consider request body size limits if you send large texts.
-   For long operations or streaming, add a WebSocket endpoint.

## 7) Troubleshooting

-   Import errors: `dreaming_hawk/__init__.py` shim adds both project root and `backend/Graphs/...` to `sys.path`. Ensure tests start from project root.
-   Missing deps: install ML libraries used by `textUtils.py` and `wordGraph.py` (e.g., `sentence_transformers`, `numpy`).
-   Port conflicts: change the Uvicorn port or use a dynamic port with `get-port` and pass it to the renderer.

## 8) Next steps

-   Flesh out `backend/app.py` to cover all methods you need.
-   Build a richer renderer (React/Vite) under `electron/renderer/`.
-   Add integration tests that hit the FastAPI endpoints for regression coverage.
