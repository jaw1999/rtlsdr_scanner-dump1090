# frequency_scanner.py

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import threading

class FrequencyScannerThread(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(object)
    signal_detected = QtCore.pyqtSignal(float, float, float)  # frequency, bandwidth, dBm
    scan_finished = QtCore.pyqtSignal()
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, sdr_controller, start_freq, stop_freq, threshold_dbm):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.threshold_dbm = threshold_dbm
        self.is_running = True

    def run(self):
        try:
            freq_step = self.sdr_controller.sample_rate / 2  # Adjust step size as needed
            frequencies = np.arange(self.start_freq, self.stop_freq, freq_step)
            for freq in frequencies:
                if not self.is_running:
                    break

                self.sdr_controller.center_freq = freq
                if not self.sdr_controller.setup():
                    self.error_occurred.emit(f"Failed to set frequency {freq/1e6:.3f} MHz")
                    continue

                # Collect samples
                samples = self.sdr_controller.read_samples()
                if samples is None:
                    continue

                # Compute power spectrum
                freq_range, power_spectrum_dbm = self.sdr_controller.compute_power_spectrum(samples)
                if freq_range is None or power_spectrum_dbm is None:
                    continue

                # Check for signals above threshold
                max_power = np.max(power_spectrum_dbm)
                if max_power > self.threshold_dbm:
                    # Estimate bandwidth
                    bandwidth = self.estimate_bandwidth(freq_range * 1e6, power_spectrum_dbm, self.threshold_dbm)
                    self.signal_detected.emit(freq, bandwidth, max_power)

                # Emit data for UI update
                self.data_ready.emit((freq_range, power_spectrum_dbm))

                # Sleep briefly to allow UI updates
                self.msleep(100)  # Adjust as needed for smoother updates
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.scan_finished.emit()

    def estimate_bandwidth(self, freq_range_hz, power_spectrum_dbm, threshold_dbm):
        above_threshold = power_spectrum_dbm > threshold_dbm
        if np.any(above_threshold):
            bandwidth = freq_range_hz[above_threshold][-1] - freq_range_hz[above_threshold][0]
        else:
            bandwidth = 0
        return bandwidth

    def stop(self):
        self.is_running = False
        self.wait()

class FrequencyScannerTab(QtWidgets.QWidget):
    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.is_scanning = False
        self.scanner_thread = None

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        self.setWindowTitle("Frequency Scanner")

        main_layout = QtWidgets.QVBoxLayout(self)

        # Control Panel
        control_panel = QtWidgets.QHBoxLayout()

        # Start Frequency Input
        self.start_freq_input = QtWidgets.QDoubleSpinBox()
        self.start_freq_input.setRange(24, 1766)
        self.start_freq_input.setDecimals(3)
        self.start_freq_input.setSuffix(" MHz")
        self.start_freq_input.setValue(100)
        control_panel.addWidget(QtWidgets.QLabel("Start Frequency:"))
        control_panel.addWidget(self.start_freq_input)

        # Stop Frequency Input
        self.stop_freq_input = QtWidgets.QDoubleSpinBox()
        self.stop_freq_input.setRange(24, 1766)
        self.stop_freq_input.setDecimals(3)
        self.stop_freq_input.setSuffix(" MHz")
        self.stop_freq_input.setValue(110)
        control_panel.addWidget(QtWidgets.QLabel("Stop Frequency:"))
        control_panel.addWidget(self.stop_freq_input)

        # Threshold Input
        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setRange(-120, 0)
        self.threshold_input.setDecimals(1)
        self.threshold_input.setSuffix(" dBm")
        self.threshold_input.setValue(-80)
        control_panel.addWidget(QtWidgets.QLabel("Threshold (dBm):"))
        control_panel.addWidget(self.threshold_input)

        # Start/Stop Button
        self.start_button = QtWidgets.QPushButton("Start Scanning")
        self.start_button.clicked.connect(self.toggle_scanning)
        control_panel.addWidget(self.start_button)

        main_layout.addLayout(control_panel)

        # Status Label
        self.status_label = QtWidgets.QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

        # Spectrum Plot
        self.spectrum_plot = pg.PlotWidget(title="Spectrum")
        self.spectrum_plot.setLabel('left', "Power (dBm)")
        self.spectrum_plot.setLabel('bottom', "Frequency (MHz)")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(color='#00CED1', width=2))
        self.spectrum_plot.setBackground('#1E1E1E')
        main_layout.addWidget(self.spectrum_plot)

        # Detected Signals Table
        self.detected_signals_table = QtWidgets.QTableWidget()
        self.detected_signals_table.setColumnCount(3)
        self.detected_signals_table.setHorizontalHeaderLabels(["Frequency (MHz)", "Bandwidth (kHz)", "Power (dBm)"])
        self.detected_signals_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(QtWidgets.QLabel("Detected Signals:"))
        main_layout.addWidget(self.detected_signals_table)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
                font-family: Arial;
                font-size: 12px;
            }
            QPushButton {
                background-color: #007ACC;
                color: #FFFFFF;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005F9E;
            }
            QDoubleSpinBox, QLabel {
                padding: 5px;
                color: #FFFFFF;
            }
            QTableWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3A3A3A;
            }
            QHeaderView::section {
                background-color: #3A3A3A;
                padding: 4px;
                border: none;
            }
        """)
        self.status_label.setStyleSheet("color: #00FF7F; font-weight: bold;")
        self.spectrum_plot.getAxis('left').setPen(pg.mkPen(color='#FFFFFF'))
        self.spectrum_plot.getAxis('bottom').setPen(pg.mkPen(color='#FFFFFF'))

    def toggle_scanning(self):
        if self.is_scanning:
            self.stop_scanning()
        else:
            self.start_scanning()

    def start_scanning(self):
        start_freq = self.start_freq_input.value() * 1e6
        stop_freq = self.stop_freq_input.value() * 1e6
        threshold_dbm = self.threshold_input.value()

        if start_freq >= stop_freq:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Start frequency must be less than stop frequency.")
            return

        self.is_scanning = True
        self.start_button.setText("Stop Scanning")
        self.detected_signals_table.setRowCount(0)
        self.status_label.setText("Status: Scanning...")

        self.scanner_thread = FrequencyScannerThread(
            self.sdr_controller,
            start_freq,
            stop_freq,
            threshold_dbm
        )

        # Connect signals
        self.scanner_thread.data_ready.connect(self.update_plots)
        self.scanner_thread.signal_detected.connect(self.add_detected_signal)
        self.scanner_thread.scan_finished.connect(self.on_scan_finished)
        self.scanner_thread.error_occurred.connect(self.show_error_message)

        self.scanner_thread.start()

    def stop_scanning(self):
        if self.scanner_thread:
            self.scanner_thread.stop()
            self.scanner_thread = None
        self.is_scanning = False
        self.start_button.setText("Start Scanning")
        self.status_label.setText("Status: Stopped")

    def update_plots(self, data):
        freq_range, power_spectrum_dbm = data
        self.spectrum_curve.setData(freq_range, power_spectrum_dbm)
        QtWidgets.QApplication.processEvents()  # Force UI update

    def add_detected_signal(self, frequency, bandwidth, power_dbm):
        row_position = self.detected_signals_table.rowCount()
        self.detected_signals_table.insertRow(row_position)
        self.detected_signals_table.setItem(row_position, 0, QtWidgets.QTableWidgetItem(f"{frequency/1e6:.3f}"))
        self.detected_signals_table.setItem(row_position, 1, QtWidgets.QTableWidgetItem(f"{bandwidth/1e3:.2f}"))
        self.detected_signals_table.setItem(row_position, 2, QtWidgets.QTableWidgetItem(f"{power_dbm:.2f}"))
        self.status_label.setText(f"Status: Signal Detected at {frequency/1e6:.3f} MHz")

    def on_scan_finished(self):
        self.is_scanning = False
        self.start_button.setText("Start Scanning")
        self.status_label.setText("Status: Scan Finished")
        QtWidgets.QMessageBox.information(self, "Scan Finished", "Frequency scan completed.")

    def show_error_message(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", message)
        self.stop_scanning()

    def closeEvent(self, event):
        self.stop_scanning()
        event.accept()
