# dialogs.py

from PyQt5 import QtWidgets

class TuneDialog(QtWidgets.QDialog):
    """Dialog to set SDR tuning parameters."""

    def __init__(self, sdr_controller, parent=None):
        super().__init__(parent)
        self.sdr_controller = sdr_controller
        self.setWindowTitle("Tune SDR")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #3B4252;
                color: #ECEFF4;
            }
            QLabel {
                color: #ECEFF4;
            }
            QPushButton {
                background-color: #81A1C1;
                color: #ECEFF4;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #88C0D0;
            }
            QLineEdit, QSpinBox {
                background-color: #4C566A;
                color: #ECEFF4;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(self)

        # Center Frequency
        freq_layout = QtWidgets.QHBoxLayout()
        freq_label = QtWidgets.QLabel("Center Frequency (MHz):")
        self.freq_input = QtWidgets.QLineEdit()
        self.freq_input.setText(str(self.sdr_controller.center_freq / 1e6))
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)
        layout.addLayout(freq_layout)

        # Gain
        gain_layout = QtWidgets.QHBoxLayout()
        gain_label = QtWidgets.QLabel("Gain (dB):")
        self.gain_input = QtWidgets.QSpinBox()
        self.gain_input.setRange(0, 50)
        self.gain_input.setValue(int(self.sdr_controller.gain) if self.sdr_controller.gain else 20)
        gain_layout.addWidget(gain_label)
        gain_layout.addWidget(self.gain_input)
        layout.addLayout(gain_layout)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def get_values(self):
        """Retrieve user-input values."""
        try:
            freq = float(self.freq_input.text()) * 1e6  # Convert MHz to Hz
        except ValueError:
            freq = self.sdr_controller.center_freq
        gain = self.gain_input.value()
        return freq, gain
