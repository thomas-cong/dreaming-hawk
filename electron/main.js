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
    const backendBase = "http://127.0.0.1:5179";
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

    if (isDev) {
        // Load from Vite dev server
        await win.loadURL("http://localhost:5173");
        win.webContents.openDevTools();
    } else {
        // Load from built files
        await win.loadFile(
            path.join(__dirname, "renderer", "dist", "index.html")
        );
    }

    // Expose backend URL to renderer
    win.webContents.executeJavaScript(
        `window.__BACKEND_URL__ = ${JSON.stringify(backendBase)};`
    );
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
});
