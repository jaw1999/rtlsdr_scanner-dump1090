# home_tab.py
from PyQt5 import QtWidgets, QtCore, QtGui
from spectrum_widget import SpectrumWidget
from waterfall_widget import WaterfallWidget
import logging

class HomeTab(QtWidgets.QWidget):
    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.initUI()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Configure logging
        logging.basicConfig(filename='home_tab.log', level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Spectrum and Waterfall container
        display_container = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_container)
        display_container.setStyleSheet("background-color: #2E3440; border-radius: 10px;")

        # Spectrum plot
        self.spectrum_widget = SpectrumWidget()
        display_layout.addWidget(self.spectrum_widget, 3)

        # Waterfall plot
        self.waterfall_widget = WaterfallWidget()
        display_layout.addWidget(self.waterfall_widget, 2)

        layout.addWidget(display_container)

        # Settings and Control panel
        settings_panel = QtWidgets.QWidget()
        settings_layout = QtWidgets.QGridLayout(settings_panel)
        settings_panel.setStyleSheet("""
            QWidget {
                background-color: #3B4252;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: #ECEFF4;
            }
            QLineEdit, QComboBox {
                background-color: #4C566A;
                color: #ECEFF4;
                border: none;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #5E81AC;
                color: #ECEFF4;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
        """)

        # Frequency settings
        settings_layout.addWidget(QtWidgets.QLabel("Center Frequency (MHz):"), 0, 0)
        self.freq_input = QtWidgets.QLineEdit(str(self.sdr_controller.center_freq / 1e6))
        settings_layout.addWidget(self.freq_input, 0, 1)

        # Sample rate settings
        settings_layout.addWidget(QtWidgets.QLabel("Sample Rate (MHz):"), 1, 0)
        self.sample_rate_input = QtWidgets.QLineEdit(str(self.sdr_controller.sample_rate / 1e6))
        settings_layout.addWidget(self.sample_rate_input, 1, 1)

        # Gain settings
        settings_layout.addWidget(QtWidgets.QLabel("Gain:"), 2, 0)
        self.gain_combo = QtWidgets.QComboBox()
        self.update_gain_combo()
        settings_layout.addWidget(self.gain_combo, 2, 1)

        # FFT size settings
        settings_layout.addWidget(QtWidgets.QLabel("FFT Size:"), 3, 0)
        self.fft_size_combo = QtWidgets.QComboBox()
        self.fft_size_combo.addItems(['512', '1024', '2048', '4096'])
        self.fft_size_combo.setCurrentText(str(self.sdr_controller.fft_size))
        settings_layout.addWidget(self.fft_size_combo, 3, 1)

        # Averaging settings
        settings_layout.addWidget(QtWidgets.QLabel("Averaging:"), 4, 0)
        self.averaging_input = QtWidgets.QLineEdit(str(self.sdr_controller.averaging))
        settings_layout.addWidget(self.averaging_input, 4, 1)

        # Color scheme settings
        settings_layout.addWidget(QtWidgets.QLabel("Color Scheme:"), 0, 2)
        self.color_scheme_combo = QtWidgets.QComboBox()
        self.color_scheme_combo.addItems(['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'])
        settings_layout.addWidget(self.color_scheme_combo, 0, 3)
        self.color_scheme_combo.currentTextChanged.connect(self.update_color_scheme)

        # Waterfall speed settings
        settings_layout.addWidget(QtWidgets.QLabel("Waterfall Speed:"), 1, 2)
        self.waterfall_speed_input = QtWidgets.QLineEdit(str(self.waterfall_widget.max_history))
        settings_layout.addWidget(self.waterfall_speed_input, 1, 3)

        # Reset SDR button
        self.reset_button = QtWidgets.QPushButton("Reset SDR")
        self.reset_button.clicked.connect(self.reset_sdr)
        settings_layout.addWidget(self.reset_button, 5, 0, 1, 2)

        # Start/Stop button
        self.start_stop_button = QtWidgets.QPushButton("Start")
        self.start_stop_button.clicked.connect(self.toggle_scan)
        settings_layout.addWidget(self.start_stop_button, 5, 2, 1, 2)

        # Status label
        self.status_label = QtWidgets.QLabel("Stopped")
        self.status_label.setStyleSheet("color: #EBCB8B;")  # Light yellow color
        settings_layout.addWidget(self.status_label, 6, 0, 1, 4)

        layout.addWidget(settings_panel)

    def update_gain_combo(self):
        current_gain = self.gain_combo.currentText()
        self.gain_combo.clear()
        self.gain_combo.addItem('auto')
        gains = self.sdr_controller.get_available_gains()
        self.gain_combo.addItems([str(gain) for gain in gains])
        index = self.gain_combo.findText(current_gain)
        if index >= 0:
            self.gain_combo.setCurrentIndex(index)
        else:
            self.gain_combo.setCurrentIndex(0)

    def toggle_scan(self):
        if self.timer.isActive():
            self.stop_scan()
        else:
            self.start_scan()

    def start_scan(self):
        try:
            if self.sdr_controller.is_active:
                self.status_label.setText("SDR is being used by another component.")
                self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
                print("SDR is active. Cannot start scan.")
                return

            # Validate and set SDR parameters
            center_freq = float(self.freq_input.text()) * 1e6
            sample_rate = float(self.sample_rate_input.text()) * 1e6
            gain = self.gain_combo.currentText()
            fft_size = int(self.fft_size_combo.currentText())
            averaging = int(self.averaging_input.text())
            waterfall_speed = int(self.waterfall_speed_input.text())

            # Additional validations
            if not (1e6 <= center_freq <= 1e9):
                raise ValueError("Center frequency must be between 1 MHz and 1 GHz.")
            if not (1e6 <= sample_rate <= 3.2e6):
                raise ValueError("Sample rate must be between 1 MHz and 3.2 MHz.")
            if fft_size not in [512, 1024, 2048, 4096]:
                raise ValueError("FFT size must be one of 512, 1024, 2048, or 4096.")
            if averaging < 1:
                raise ValueError("Averaging must be at least 1.")
            if waterfall_speed < 1:
                raise ValueError("Waterfall speed must be at least 1.")

            # Assign validated values
            self.sdr_controller.center_freq = center_freq
            self.sdr_controller.sample_rate = sample_rate
            self.sdr_controller.gain = gain
            self.sdr_controller.fft_size = fft_size
            self.sdr_controller.averaging = averaging
            self.waterfall_widget.max_history = waterfall_speed

            # Setup SDR
            if self.sdr_controller.setup():
                self.timer.start(100)  # Update every 100 ms
                self.start_stop_button.setText("Stop")
                self.status_label.setText("Scanning")
                self.status_label.setStyleSheet("color: #A3BE8C;")  # Light green color
                logging.info("Started SDR scan.")
                print("SDR scan started.")
            else:
                self.status_label.setText("Failed to set up SDR")
                self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
                logging.error("Failed to set up SDR.")
                print("Failed to set up SDR.")
        except ValueError as ve:
            self.status_label.setText(str(ve))
            self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
            logging.error(f"Input validation error: {str(ve)}")
            print(f"Input validation error: {str(ve)}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
            logging.error(f"Error starting SDR scan: {str(e)}")
            print(f"Error starting SDR scan: {str(e)}")

    def stop_scan(self):
        self.timer.stop()
        self.start_stop_button.setText("Start")
        self.sdr_controller.close()
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: #EBCB8B;")  # Light yellow color
        logging.info("Stopped SDR scan.")
        print("SDR scan stopped.")

    def reset_sdr(self):
        self.stop_scan()
        if self.sdr_controller.reset():
            self.status_label.setText("SDR Reset Successfully")
            self.status_label.setStyleSheet("color: #A3BE8C;")  # Light green color
            self.update_gain_combo()
            logging.info("SDR reset successfully.")
            print("SDR reset successfully.")
        else:
            self.status_label.setText("Failed to Reset SDR")
            self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
            logging.error("Failed to reset SDR.")
            print("Failed to reset SDR.")

    def update_plot(self):
        try:
            samples = self.sdr_controller.read_samples()
            freq_range, power_spectrum_db = self.sdr_controller.compute_power_spectrum(samples)

            if freq_range is not None and power_spectrum_db is not None:
                self.spectrum_widget.update(freq_range, power_spectrum_db)
                self.waterfall_widget.update(power_spectrum_db)
                logging.info("Updated spectrum and waterfall plots.")
                print("Updated spectrum and waterfall plots.")
            else:
                self.status_label.setText("Invalid data received")
                self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
                logging.error("Invalid data received while updating plots.")
                print("Invalid data received while updating plots.")
        except Exception as e:
            self.status_label.setText(f"Error updating plot: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
            self.stop_scan()
            logging.error(f"Error updating plot: {str(e)}")
            print(f"Error updating plot: {str(e)}")

    def update_color_scheme(self, scheme):
        try:
            self.spectrum_widget.set_color_scheme(scheme)
            self.waterfall_widget.set_color_scheme(scheme)
            logging.info(f"Updated color scheme to {scheme}.")
            print(f"Updated color scheme to {scheme}.")
        except Exception as e:
            self.status_label.setText(f"Error updating color scheme: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
            logging.error(f"Error updating color scheme: {str(e)}")
            print(f"Error updating color scheme: {str(e)}")

    def disable_sdr_controls(self):
        self.start_stop_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.freq_input.setEnabled(False)
        self.sample_rate_input.setEnabled(False)
        self.gain_combo.setEnabled(False)
        self.fft_size_combo.setEnabled(False)
        self.averaging_input.setEnabled(False)
        self.color_scheme_combo.setEnabled(False)
        self.waterfall_speed_input.setEnabled(False)
        self.status_label.setText("Dump1090 is running. SDR controls disabled.")
        self.status_label.setStyleSheet("color: #BF616A;")  # Light red color
        logging.info("SDR controls disabled due to Dump1090 running.")
        print("SDR controls disabled due to Dump1090 running.")

    def enable_sdr_controls(self):
        self.start_stop_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.freq_input.setEnabled(True)
        self.sample_rate_input.setEnabled(True)
        self.gain_combo.setEnabled(True)
        self.fft_size_combo.setEnabled(True)
        self.averaging_input.setEnabled(True)
        self.color_scheme_combo.setEnabled(True)
        self.waterfall_speed_input.setEnabled(True)
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: #EBCB8B;")  # Light yellow color
        logging.info("SDR controls enabled after Dump1090 stopped.")
        print("SDR controls enabled after Dump1090 stopped.")

    def closeEvent(self, event):
        self.stop_scan()
        super().closeEvent(event)
