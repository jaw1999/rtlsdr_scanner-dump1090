import pyqtgraph as pg
from PyQt5 import QtGui

class SpectrumWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground('#2E3440')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('left', 'Power', units='dB', color='#ECEFF4')
        self.setLabel('bottom', 'Frequency', units='MHz', color='#ECEFF4')
        self.setTitle("Spectrum", color='#ECEFF4')
        
        self.plot_curve = self.plot(pen=pg.mkPen(color='#88C0D0', width=2))
        
        # Add a text item for peak frequency
        self.peak_text = pg.TextItem(anchor=(1, 1), color='#EBCB8B')
        self.addItem(self.peak_text)

    def update(self, freq_range, power_spectrum_db):
        self.plot_curve.setData(freq_range, power_spectrum_db)
        
        # Find and display peak frequency
        peak_idx = power_spectrum_db.argmax()
        peak_freq = freq_range[peak_idx]
        peak_power = power_spectrum_db[peak_idx]
        self.peak_text.setText(f"Peak: {peak_freq:.2f} MHz, {peak_power:.2f} dB")
        self.peak_text.setPos(freq_range[-1], max(power_spectrum_db))