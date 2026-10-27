const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFile: () => ipcRenderer.invoke('dialog:openFile'),
  uploadFile: (filePath) => ipcRenderer.invoke('api:uploadFile', filePath),
  openPath: (pathStr) => ipcRenderer.invoke('app:openPath', pathStr),
  setTheme: (theme) => ipcRenderer.invoke('app:setTheme', theme)
});
