# spectrum_lora_widget.py
from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np

class SpectrumLoRaWidget(QtWidgets.QWidget):    # Changed from LoRaSpectrumWidget to match the import
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.peak_hold_enabled = False
        self.peak_hold_data = None

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#2E3440')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Power', units='dB', color='#ECEFF4')
        self.plot_widget.setLabel('bottom', 'Frequency', units='MHz', color='#ECEFF4')
        self.plot_widget.setTitle('Spectrum', color='#ECEFF4')
        self.plot_widget.setYRange(-150, -50)

        # Add plot curve
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen(color='#88C0D0', width=2))
        self.peak_hold_curve = self.plot_widget.plot(pen=pg.mkPen(color='#BF616A', width=2))

        # Add peak text item
        self.peak_text = pg.TextItem(anchor=(1, 1), color='#EBCB8B')
        self.plot_widget.addItem(self.peak_text)
        
        layout.addWidget(self.plot_widget)

    def update(self, freq_range, power_spectrum_db):
        try:
            if self.peak_hold_enabled:
                if self.peak_hold_data is None:
                    self.peak_hold_data = power_spectrum_db.copy()
                else:
                    self.peak_hold_data = np.maximum(self.peak_hold_data, power_spectrum_db)
                self.peak_hold_curve.setData(freq_range, self.peak_hold_data)

            self.plot_curve.setData(freq_range, power_spectrum_db)

            # Find and display peak
            peak_idx = np.argmax(power_spectrum_db)
            peak_freq = freq_range[peak_idx]
            peak_power = power_spectrum_db[peak_idx]
            self.peak_text.setText(f"Peak: {peak_freq:.2f} MHz, {peak_power:.2f} dB")
            self.peak_text.setPos(freq_range[-1], max(power_spectrum_db))

        except Exception as e:
            print(f"Error updating spectrum: {str(e)}")

    def set_peak_hold(self, enabled):
        self.peak_hold_enabled = enabled
        if not enabled:
            self.peak_hold_data = None
            self.peak_hold_curve.clear()

    def clear(self):
        self.plot_curve.clear()
        self.peak_hold_curve.clear()
        self.peak_hold_data = None
        self.peak_text.setText("")