import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import logging

class WaterfallLoRaWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.max_history = 100  # Number of lines to keep in waterfall
        self.history = None
        self._setup_default_colormap()
        
    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create graphics view and plot
        self.graphicsView = pg.GraphicsView()
        self.graphicsView.setBackground('#2E3440')
        self.layout.addWidget(self.graphicsView)
        
        # Setup plot
        self.plot = pg.PlotItem()
        self.graphicsView.setCentralItem(self.plot)
        
        # Setup image item
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        
        # Remove axes for cleaner look
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')

    def _setup_default_colormap(self):
        """Initialize the default color scheme (viridis-like)"""
        self.color_schemes = {
            'viridis': [
                (0, 0, 0),      # Black
                (68, 1, 84),    # Dark purple
                (59, 82, 139),  # Blue
                (33, 145, 140), # Cyan
                (94, 201, 98),  # Green
                (253, 231, 37)  # Yellow
            ],
            'plasma': [
                (0, 0, 0),
                (32, 5, 103),
                (140, 14, 164),
                (229, 67, 97),
                (254, 180, 84),
                (252, 253, 191)
            ],
            'magma': [
                (0, 0, 0),
                (28, 16, 68),
                (117, 15, 109),
                (213, 48, 77),
                (249, 142, 82),
                (252, 253, 191)
            ],
            'inferno': [
                (0, 0, 0),
                (20, 11, 52),
                (120, 28, 109),
                (218, 60, 67),
                (252, 172, 83),
                (252, 255, 164)
            ]
        }
        
        # Set default colormap (viridis)
        self._set_colormap('viridis')

    def _set_colormap(self, scheme_name):
        """Set up color lookup table for the given scheme"""
        try:
            if scheme_name.lower() in self.color_schemes:
                colors = self.color_schemes[scheme_name.lower()]
                positions = np.linspace(0.0, 1.0, len(colors))
                cmap = pg.ColorMap(pos=positions, color=colors)
                self.lut = cmap.getLookupTable(0.0, 1.0, 256)
                if self.history is not None:
                    self.img.setLookupTable(self.lut)
        except Exception as e:
            logging.error(f"Error setting colormap: {str(e)}")

    def update(self, spectrum):
        """Update the waterfall display with new spectrum data"""
        try:
            if spectrum is None:
                return

            # Initialize history if needed
            if self.history is None:
                self.history = np.zeros((self.max_history, len(spectrum)))
            
            # Roll the history array and add new data
            self.history = np.roll(self.history, 1, axis=0)
            self.history[0] = spectrum
            
            # Normalize the data for LoRa specific range
            min_val = -150  # minimum dBm
            max_val = -50   # maximum dBm
            normalized = np.clip((self.history - min_val) / (max_val - min_val), 0, 1)
            
            # Convert to 8-bit color
            colored = (normalized * 255).astype(np.uint8)
            
            # Transpose and update image
            colored = np.transpose(colored)
            self.img.setImage(colored, autoLevels=False)
            self.img.setLookupTable(self.lut)
            
            # Set correct scale and position
            self.img.resetTransform()
            self.img.scale(1, 1)
            self.img.setPos(0, 0)
            
            # Flip vertically for correct orientation
            self.img.setTransform(QtGui.QTransform().scale(1, -1))
            
        except Exception as e:
            logging.error(f"Error updating waterfall: {str(e)}")

    def set_color_scheme(self, scheme):
        """Set the color scheme for the waterfall display"""
        try:
            scheme = scheme.lower()
            if scheme in self.color_schemes:
                self._set_colormap(scheme)
            else:
                logging.warning(f"Unknown color scheme: {scheme}, using default")
                self._set_colormap('viridis')
        except Exception as e:
            logging.error(f"Error setting color scheme: {str(e)}")

    def set_history_size(self, size):
        """Set the number of history lines to display"""
        try:
            if size != self.max_history:
                self.max_history = size
                self.clear_data()
        except Exception as e:
            logging.error(f"Error setting history size: {str(e)}")

    def clear(self):
        """Clear the waterfall display"""
        try:
            self.history = None
            if self.img is not None:
                self.img.clear()
        except Exception as e:
            logging.error(f"Error clearing waterfall: {str(e)}")

    def clear_data(self):
        """Alias for clear() method to maintain consistency with spectrum widget"""
        self.clear()