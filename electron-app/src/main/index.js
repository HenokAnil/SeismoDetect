const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1300,
    height: 850,
    minWidth: 1000,
    minHeight: 700,
    title: "SeismoDetect - Seismic Phase Detection & Vector Intelligence",
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  
  // Uncomment below to debug in DevTools if desired
  // mainWindow.webContents.openDevTools();
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// IPC Handler for file open dialog
ipcMain.handle('dialog:openFile', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'Seismic Data (*.mseed)', extensions: ['mseed', 'miniseed'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  if (canceled || !filePaths || filePaths.length === 0) {
    return null;
  }
  return filePaths[0];
});

// IPC Handler to upload a local file to the FastAPI backend
ipcMain.handle('api:uploadFile', async (event, filePath) => {
  try {
    if (!fs.existsSync(filePath)) {
      return { success: false, error: "Selected file does not exist on disk." };
    }

    const fileBuffer = fs.readFileSync(filePath);
    const fileName = path.basename(filePath);

    // Native Node Blob & FormData
    const blob = new Blob([fileBuffer]);
    const formData = new FormData();
    formData.append('file', blob, fileName);

    const response = await fetch('http://127.0.0.1:8000/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errText = await response.text();
      return { success: false, error: `Upload failed (Status ${response.status}): ${errText}` };
    }

    const result = await response.json();
    return { success: true, data: result };
  } catch (err) {
    const detail = err.cause ? ` (${err.cause.message || err.cause})` : '';
    return { 
      success: false, 
      error: `${err.message}${detail}. Ensure the FastAPI backend is running and ready on http://127.0.0.1:8000.` 
    };
  }
});

// IPC Handler to open external path or PDF
ipcMain.handle('app:openPath', async (event, targetPath) => {
  if (targetPath) {
    shell.openPath(targetPath);
  }
});
