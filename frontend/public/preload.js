// public/preload.js
const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe, world-isolated API to the React app (renderer process)
contextBridge.exposeInMainWorld(
  'electronAPI', // This will be window.electronAPI in your React app
  {
    // Example function that React can call
    // This sends a message to the main process and waits for a response
    sendMessage: (message) => ipcRenderer.invoke('my-invokable-channel', message),

    // You can also expose functions that listen for messages *from* the main process
    // ipcRenderer.on('some-channel', (event, ...args) => { ... });
  }
);
