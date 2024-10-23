# classification_waterfall_widget.py

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import logging

class ClassificationWaterfallWidget(QtWidgets.QWidget):
    """Waterfall widget for signal classification."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize attributes before setting up the UI
        self.max_buffer_size = 100  # Maximum number of lines in the waterfall
        self.data_buffer = []
        self.initialized = False  # Flag to check if plot is initialized
        
        self.setup_ui()
        logging.info("ClassificationWaterfallWidget initialized with max_buffer_size=100.")
    
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(facecolor='#2E3440')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # Ensure canvas expands
        layout.addWidget(self.canvas)
        
        # Setup axis
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2E3440')
        self.setup_plot_appearance()

    def setup_plot_appearance(self):
        """Configure the plot appearance."""
        self.ax.grid(True, color='#4C566A', alpha=0.5)
        self.ax.set_title("Waterfall Display", color='#ECEFF4', pad=10)
        self.ax.set_xlabel("Frequency (MHz)", color='#ECEFF4')
        self.ax.set_ylabel("Time", color='#ECEFF4')
        self.ax.tick_params(axis='both', colors='#ECEFF4')
        self.ax.set_ylim(0, self.max_buffer_size)
        self.ax.invert_yaxis()  # Newest data on top

    def setup_plot(self, frequencies, power_spectrum):
        """Initialize the waterfall plot with the first set of data."""
        # Normalize power spectrum to [0, 1] for colormap (optional)
        power_min = power_spectrum.min()
        power_max = power_spectrum.max()
        if power_max - power_min == 0:
            norm_powers = power_spectrum
            logging.warning("Power spectrum range is zero. Skipping normalization.")
        else:
            norm_powers = (power_spectrum - power_min) / (power_max - power_min)
        
        # Initialize data buffer
        self.data_buffer.append(norm_powers)
        initial_data = np.array(self.data_buffer)
        
        # Set extent based on actual frequencies
        self.frequency_min = frequencies.min()
        self.frequency_max = frequencies.max()
        
        # Initialize imshow
        self.im = self.ax.imshow(
            initial_data,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            extent=[self.frequency_min, self.frequency_max, 0, self.max_buffer_size],
            origin='lower'
        )
        self.figure.tight_layout()
        self.canvas.draw()
        logging.info("Waterfall plot initialized with first data set.")
        self.initialized = True

    def update_plot(self, frequencies, power_spectrum):
        """
        Update the waterfall plot with new power spectrum data.

        Parameters:
        - frequencies: 1D numpy array of frequency values (MHz)
        - power_spectrum: 1D numpy array of power values (dBm)
        """
        try:
            if len(frequencies) != len(power_spectrum):
                logging.error("Frequencies and power_spectrum must have the same length.")
                return
            
            # Initialize plot with first data set if not already done
            if not self.initialized:
                self.setup_plot(frequencies, power_spectrum)
                return
            
            # Check if frequency range has changed
            if frequencies.min() != self.frequency_min or frequencies.max() != self.frequency_max:
                logging.warning("Frequency range has changed. Reinitializing plot.")
                self.ax.clear()
                self.setup_plot(frequencies, power_spectrum)
                return
            
            # Normalize power spectrum to [0, 1] for colormap (optional)
            power_min = power_spectrum.min()
            power_max = power_spectrum.max()
            if power_max - power_min == 0:
                norm_powers = power_spectrum
                logging.warning("Power spectrum range is zero. Skipping normalization.")
            else:
                norm_powers = (power_spectrum - power_min) / (power_max - power_min)
            
            # Append normalized power spectrum to data buffer
            self.data_buffer.append(norm_powers)
            if len(self.data_buffer) > self.max_buffer_size:
                self.data_buffer.pop(0)
            
            # Convert buffer to 2D array for imshow
            data = np.array(self.data_buffer)
            # Ensure consistent dimensions
            if data.shape[1] != len(frequencies):
                logging.error("Mismatch between buffer data and current frequency length.")
                return
            
            # Update the image data
            self.im.set_data(data)
            self.im.set_extent([self.frequency_min, self.frequency_max, 0, self.max_buffer_size])
            
            # Adjust y-axis (time axis)
            self.ax.set_ylim(0, self.max_buffer_size)
            
            self.canvas.draw_idle()
            logging.debug("Waterfall plot updated successfully.")
        except Exception as e:
            logging.error(f"Error updating waterfall plot: {e}")

    def get_selected_data(self):
        """
        Retrieve the data from the selected region.

        Returns:
        - freqs: 1D numpy array of frequency values in the selected region
        - powers: 1D numpy array of power values in the selected region
        """
        # Implementation depends on how selection is handled in ClassificationSpectrumWidget
        # Placeholder implementation
        return None, None

    def clear_selection(self):
        """Clear any region selection."""
        # Implementation depends on how selection is handled in ClassificationSpectrumWidget
        pass
