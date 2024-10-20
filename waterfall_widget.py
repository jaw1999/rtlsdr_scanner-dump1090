import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

class WaterfallWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.graphicsView = pg.GraphicsView()
        self.layout.addWidget(self.graphicsView)
        
        self.plot = pg.PlotItem()
        self.graphicsView.setCentralItem(self.plot)
        
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)

        # Remove axes
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')

        # Set up color lookup table
        colors = [
            (0, 0, 0),        # Black
            (0, 0, 255),      # Blue
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Yellow
            (255, 0, 0)       # Red
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)
        
        self.history = None
        self.max_history = 100  # Number of lines to keep in the waterfall

    def update(self, spectrum):
        if self.history is None:
            self.history = np.zeros((self.max_history, len(spectrum)))
        
        # Roll the history array and add new data
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = spectrum
        
        # Normalize the data
        normalized = (self.history - np.min(self.history)) / (np.max(self.history) - np.min(self.history) + 1e-6)
        
        # Convert to 8-bit color
        colored = (normalized * 255).astype(np.uint8)
        
        # Transpose the image to make it flow from top to bottom
        colored = np.transpose(colored)
        
        self.img.setImage(colored, autoLevels=False)
        self.img.setLookupTable(self.lut)
        
        # Set the correct scale and position
        self.img.resetTransform()
        self.img.scale(1, 1)
        self.img.setPos(0, 0)

        # Flip the image vertically
        self.img.setTransform(QtGui.QTransform().scale(1, -1))