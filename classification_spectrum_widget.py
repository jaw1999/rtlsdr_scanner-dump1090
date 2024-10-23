# classification_spectrum_widget.py
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import logging

class ClassificationSpectrumWidget(QtWidgets.QWidget):
    """Spectrum widget with region selection capabilities for signal classification."""
    
    # Custom signals
    region_selected = QtCore.pyqtSignal(float, float)  # Emits start and end frequencies
    region_changed = QtCore.pyqtSignal(float, float)   # Emits during drag
    selection_cleared = QtCore.pyqtSignal()            # Emits when selection is cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_variables()
        self.connect_events()

    def setup_ui(self):
        """Initialize the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(facecolor='#2E3440')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Setup axis
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2E3440')
        self.setup_plot_appearance()
        
        # Create the spectrum line
        self.spectrum_line, = self.ax.plot([], [], color='#88C0D0', linewidth=1)
        
        self.figure.tight_layout()

    def setup_variables(self):
        """Initialize internal variables."""
        self.selecting = False
        self.start_freq = None
        self.current_freq = None
        self.selection_rect = None
        self.freq_range = None
        self.power_data = None
        self.last_power_data = None

    def setup_plot_appearance(self):
        """Configure the plot appearance."""
        self.ax.grid(True, color='#4C566A', alpha=0.5)
        self.ax.set_title("Signal Classification Spectrum", color='#ECEFF4', pad=10)
        self.ax.set_xlabel("Frequency (MHz)", color='#ECEFF4')
        self.ax.set_ylabel("Power (dBm)", color='#ECEFF4')
        self.ax.tick_params(axis='both', colors='#ECEFF4')
        
        # Set y-axis limits for better visualization
        self.ax.set_ylim(-100, 0)

    def connect_events(self):
        """Connect mouse and keyboard events for selection."""
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def update_plot(self, frequencies, powers):
        """Update the spectrum plot with new data."""
        if len(frequencies) != len(powers):
            logging.error("Frequencies and powers must have the same length.")
            return
            
        self.freq_range = frequencies
        self.power_data = powers
        self.spectrum_line.set_data(frequencies, powers)
        
        # Update axis limits if needed
        self.ax.set_xlim(frequencies.min(), frequencies.max())
        
        # Dynamically adjust y-axis based on power data
        self.ax.set_ylim(min(powers) - 10, max(powers) + 10)
        
        # Redraw selection if it exists
        if self.selection_rect:
            self.redraw_selection()
            
        self.canvas.draw()
        self.last_power_data = powers.copy()
        logging.info("Spectrum plot updated successfully.")

    def on_mouse_press(self, event):
        """Handle mouse press events for selection."""
        if event.inaxes != self.ax or event.button != 1:
            return
            
        # Clear previous selection
        self.clear_selection()
        
        self.selecting = True
        self.start_freq = event.xdata
        self.current_freq = event.xdata
        
        # Create initial selection rectangle
        self.selection_rect = Rectangle(
            (self.start_freq, self.ax.get_ylim()[0]),
            0,  # Width will be updated during mouse move
            self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
            facecolor='#EBCB8B',
            edgecolor='#D08770',
            alpha=0.3
        )
        self.ax.add_patch(self.selection_rect)
        self.canvas.draw()
        logging.debug(f"Selection started at {self.start_freq} MHz.")

    def on_mouse_move(self, event):
        """Handle mouse movement during selection."""
        if not self.selecting or event.inaxes != self.ax:
            return
            
        self.current_freq = event.xdata
        self.redraw_selection()
        self.region_changed.emit(
            min(self.start_freq, self.current_freq),
            max(self.start_freq, self.current_freq)
        )
        logging.debug(f"Selection changed to {min(self.start_freq, self.current_freq)} MHz - {max(self.start_freq, self.current_freq)} MHz.")

    def on_mouse_release(self, event):
        """Handle mouse release to complete selection."""
        if not self.selecting:
            return
            
        self.selecting = False
        if self.start_freq and self.current_freq:
            start = min(self.start_freq, self.current_freq)
            end = max(self.start_freq, self.current_freq)
            self.region_selected.emit(start, end)
            logging.info(f"Region selected from {start} MHz to {end} MHz.")

    def on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'escape':
            self.clear_selection()
            self.selection_cleared.emit()
            logging.info("Selection cleared via Escape key.")

    def clear_selection(self):
        """Clear the current selection."""
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.start_freq = None
            self.current_freq = None
            self.canvas.draw()
            logging.info("Selection cleared.")

    def redraw_selection(self):
        """Redraw the selection rectangle."""
        if self.selection_rect and self.start_freq and self.current_freq:
            x = min(self.start_freq, self.current_freq)
            width = abs(self.current_freq - self.start_freq)
            self.selection_rect.set_x(x)
            self.selection_rect.set_width(width)
            self.canvas.draw()
            logging.debug(f"Selection rectangle updated: x={x}, width={width} MHz.")

    def get_selected_data(self):
        """Get the data from the selected region."""
        if not all([self.start_freq, self.current_freq, self.freq_range is not None, self.power_data is not None]):
            logging.warning("Selected region data is incomplete.")
            return None, None
            
        start = min(self.start_freq, self.current_freq)
        end = max(self.start_freq, self.current_freq)
        
        # Get indices for the selected frequency range
        mask = (self.freq_range >= start) & (self.freq_range <= end)
        selected_freqs = self.freq_range[mask]
        selected_powers = self.power_data[mask]
        
        if len(selected_freqs) == 0 or len(selected_powers) == 0:
            logging.warning("No data found in the selected frequency range.")
            return None, None
        
        logging.debug(f"Selected data points: {len(selected_freqs)}")
        return selected_freqs, selected_powers
