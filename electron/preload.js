const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
    getBackendUrl: () => window.__BACKEND_URL__,
});
