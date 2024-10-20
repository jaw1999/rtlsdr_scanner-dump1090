from PyQt5 import QtWidgets

class SettingsTab(QtWidgets.QWidget):
    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QFormLayout(self)

        self.center_freq_input = QtWidgets.QLineEdit(str(self.sdr_controller.center_freq / 1e6))
        layout.addRow("Center Frequency (MHz):", self.center_freq_input)

        self.sample_rate_input = QtWidgets.QLineEdit(str(self.sdr_controller.sample_rate / 1e6))
        layout.addRow("Sample Rate (MHz):", self.sample_rate_input)

        self.gain_input = QtWidgets.QLineEdit(str(self.sdr_controller.gain))
        layout.addRow("Gain (dB or 'auto'):", self.gain_input)

        self.apply_button = QtWidgets.QPushButton("Apply Settings")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addRow(self.apply_button)

    def apply_settings(self):
        try:
            self.sdr_controller.center_freq = float(self.center_freq_input.text()) * 1e6
            self.sdr_controller.sample_rate = float(self.sample_rate_input.text()) * 1e6
            self.sdr_controller.gain = self.gain_input.text()
            if self.sdr_controller.gain.lower() != 'auto':
                self.sdr_controller.gain = float(self.sdr_controller.gain)
            self.sdr_controller.setup()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")