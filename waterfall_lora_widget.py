# waterfall_lora_widget.py
import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

class WaterfallLoRaWidget(QtWidgets.QWidget):
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
            (0, 0, 0),      # Black
            (0, 0, 255),    # Blue
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
            (255, 0, 0)     # Red
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)
        
        # Initialize history
        self.history = None
        self.max_history = 100  # Number of lines to keep in waterfall
        
        # Set background color
        self.graphicsView.setBackground('#2E3440')

    def update(self, spectrum):
        try:
            if self.history is None:
                self.history = np.zeros((self.max_history, len(spectrum)))
            
            # Roll the history array and add new data
            self.history = np.roll(self.history, 1, axis=0)
            self.history[0] = spectrum
            
            # Normalize the data with fixed range for LoRa
            min_val = -150  # minimum dBm
            max_val = -50   # maximum dBm
            normalized = np.clip((self.history - min_val) / (max_val - min_val), 0, 1)
            
            # Convert to 8-bit color
            colored = (normalized * 255).astype(np.uint8)
            
            # Transpose the image to make it flow from top to bottom
            colored = np.transpose(colored)
            
            # Update image
            self.img.setImage(colored, autoLevels=False)
            self.img.setLookupTable(self.lut)
            
            # Set the correct scale and position
            self.img.resetTransform()
            self.img.scale(1, 1)
            self.img.setPos(0, 0)
            
            # Flip the image vertically
            self.img.setTransform(QtGui.QTransform().scale(1, -1))
            
        except Exception as e:
            print(f"Error updating waterfall: {str(e)}")

    def set_color_scheme(self, scheme):
        try:
            scheme_colors = {
                'viridis': [
                    (0, 0, 0),
                    (0, 0, 255),
                    (0, 255, 0),
                    (255, 255, 0)
                ],
                'plasma': [
                    (0, 0, 0),
                    (128, 0, 255),
                    (255, 0, 128),
                    (255, 255, 0)
                ],
                'magma': [
                    (0, 0, 0),
                    (128, 0, 128),
                    (255, 64, 64),
                    (255, 255, 255)
                ],
                'inferno': [
                    (0, 0, 0),
                    (128, 0, 0),
                    (255, 128, 0),
                    (255, 255, 0)
                ]
            }
            
            if scheme.lower() in scheme_colors:
                colors = scheme_colors[scheme.lower()]
                cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
                self.lut = cmap.getLookupTable(0.0, 1.0, 256)
                if self.history is not None:
                    self.img.setLookupTable(self.lut)
                
        except Exception as e:
            print(f"Error setting color scheme: {str(e)}")

    def set_history_size(self, size):
        if size != self.max_history:
            self.max_history = size
            self.history = None

    def clear(self):
        self.history = None
        self.img.clear()