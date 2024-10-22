from PyQt5 import QtWidgets, QtCore
import numpy as np
import logging
from spectrum_lora_widget import SpectrumLoRaWidget
from waterfall_lora_widget import WaterfallLoRaWidget
from datetime import datetime
from scipy.signal import fftconvolve
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from typing import List, Optional
import weakref

class ThreadManager:
    """Manages thread lifecycle and cleanup"""
    def __init__(self):
        self.active_threads: List[QThread] = []
        self._cleanup_lock = QtCore.QMutex()

    def add_thread(self, thread: QThread):
        with QtCore.QMutexLocker(self._cleanup_lock):
            # Connect finished signal to remove thread from list
            thread.finished.connect(lambda: self.remove_thread(thread))
            self.active_threads.append(thread)

    def remove_thread(self, thread: QThread):
        with QtCore.QMutexLocker(self._cleanup_lock):
            if thread in self.active_threads:
                self.active_threads.remove(thread)

    def stop_all(self):
        with QtCore.QMutexLocker(self._cleanup_lock):
            for thread in self.active_threads[:]:  # Create a copy of the list to iterate
                try:
                    if thread.isRunning():
                        thread.quit()
                        thread.wait(1000)  # Wait up to 1 second
                        if thread.isRunning():
                            thread.terminate()
                except RuntimeError:
                    # Thread was already deleted, remove it from our list
                    if thread in self.active_threads:
                        self.active_threads.remove(thread)
            self.active_threads.clear()

class DetectionWorker(QObject):
    detection_finished = pyqtSignal(bool, float, float, float, int, float)
    
    def __init__(self, samples, sample_rate, sf_values, bw_values, threshold_db, center_freq):
        super().__init__()
        self.samples = samples
        self.sample_rate = sample_rate
        self.sf_values = sf_values
        self.bw_values = bw_values
        self.threshold_db = threshold_db
        self.center_freq = center_freq
        self._is_running = True

    def stop(self):
        self._is_running = False

    @QtCore.pyqtSlot()
    def process(self):
        if self._is_running:
            detected, detected_freq, power_dbm, snr, sf, bw = self.detect_lora_signal()
            if self._is_running:  # Check again before emitting
                self.detection_finished.emit(detected, detected_freq, power_dbm, snr, sf, bw)

    def detect_lora_signal(self):
        try:
            detected_signals = []
            num_samples = len(self.samples)
            t = np.arange(num_samples) / self.sample_rate

            for sf in self.sf_values:
                if not self._is_running:
                    return False, 0.0, 0.0, 0.0, 0, 0.0
                    
                for bw in self.bw_values:
                    if not self._is_running:
                        return False, 0.0, 0.0, 0.0, 0, 0.0
                        
                    if self.sample_rate < bw:
                        continue  # Skip if sample rate is insufficient
                    k = bw / (2 ** sf)
                    ref_chirp = np.exp(1j * 2 * np.pi * (0.5 * k * t ** 2))
                    correlation = fftconvolve(self.samples, ref_chirp[::-1].conj(), mode='same')
                    magnitude = np.abs(correlation)
                    peak_magnitude = np.max(magnitude)
                    noise_floor = np.median(magnitude)
                    snr = 20 * np.log10(peak_magnitude / (noise_floor + 1e-9))
                    threshold = noise_floor * 10 ** (self.threshold_db / 20)
                    if peak_magnitude > threshold:
                        detected_signals.append({
                            'sf': sf,
                            'bw': bw,
                            'snr': snr,
                            'magnitude': peak_magnitude,
                            'peak_index': np.argmax(magnitude)
                        })

            if detected_signals and self._is_running:
                best_signal = max(detected_signals, key=lambda x: x['snr'])
                freq_resolution = self.sample_rate / num_samples
                freq_offset = (best_signal['peak_index'] - num_samples // 2) * freq_resolution
                detected_freq = (self.center_freq + freq_offset) / 1e6  # Convert to MHz
                power_dbm = 20 * np.log10(best_signal['magnitude']) - 30  # Approximate
                return True, detected_freq, power_dbm, best_signal['snr'], best_signal['sf'], best_signal['bw']
            else:
                return False, 0.0, 0.0, 0.0, 0, 0.0

        except Exception as e:
            logging.error(f"Detection error: {str(e)}")
            return False, 0.0, 0.0, 0.0, 0, 0.0

class LoRaTab(QtWidgets.QWidget):
    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.running = False
        self.worker: Optional[DetectionWorker] = None
        self.thread_manager = ThreadManager()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.detection_count = 0
        self.last_detection_time = None

        # Configure logging
        logging.basicConfig(filename='lora_tab.log', level=logging.INFO,
                          format='%(asctime)s:%(levelname)s:%(message)s')

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Display container
        display_container = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_container)
        display_container.setStyleSheet("background-color: #2E3440; border-radius: 10px;")

        # Add spectrum widget
        self.spectrum_widget = SpectrumLoRaWidget()
        display_layout.addWidget(self.spectrum_widget, 3)

        # Add waterfall widget
        self.waterfall_widget = WaterfallLoRaWidget()
        display_layout.addWidget(self.waterfall_widget, 2)

        display_container.setLayout(display_layout)
        layout.addWidget(display_container)

        # Horizontal layout for settings and info panels
        h_layout = QtWidgets.QHBoxLayout()

        # Settings panel
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

        row = 0

        # Frequency settings
        settings_layout.addWidget(QtWidgets.QLabel("Preset Frequencies:"), row, 0)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "433.175 MHz - EU",
            "868.100 MHz - EU",
            "868.300 MHz - EU",
            "868.500 MHz - EU",
            "915.000 MHz - US",
            "915.200 MHz - US",
            "915.400 MHz - US"
        ])
        self.preset_combo.currentTextChanged.connect(self.handle_preset_change)
        settings_layout.addWidget(self.preset_combo, row, 1)

        # Gain settings
        settings_layout.addWidget(QtWidgets.QLabel("Gain:"), row, 2)
        self.gain_combo = QtWidgets.QComboBox()
        self.update_gain_combo()
        settings_layout.addWidget(self.gain_combo, row, 3)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Center Frequency (MHz):"), row, 0)
        self.freq_input = QtWidgets.QLineEdit("868.0")
        settings_layout.addWidget(self.freq_input, row, 1)

        # Sample Rate settings
        settings_layout.addWidget(QtWidgets.QLabel("Sample Rate (MHz):"), row, 2)
        self.sample_rate_input = QtWidgets.QLineEdit("1.024")
        settings_layout.addWidget(self.sample_rate_input, row, 3)

        row += 1
        # Spreading Factor settings
        settings_layout.addWidget(QtWidgets.QLabel("Spreading Factor (SF):"), row, 0)
        self.sf_combo = QtWidgets.QComboBox()
        self.sf_combo.addItems(['All', '7', '8', '9', '10', '11', '12'])
        settings_layout.addWidget(self.sf_combo, row, 1)

        # Bandwidth settings
        settings_layout.addWidget(QtWidgets.QLabel("Bandwidth (kHz):"), row, 2)
        self.bw_combo = QtWidgets.QComboBox()
        self.bw_combo.addItems(['All', '125', '250', '500'])
        settings_layout.addWidget(self.bw_combo, row, 3)

        row += 1
        # Threshold settings
        settings_layout.addWidget(QtWidgets.QLabel("Detection Threshold (dB):"), row, 0)
        self.threshold_input = QtWidgets.QLineEdit("3")
        settings_layout.addWidget(self.threshold_input, row, 1)

        # FFT Size settings
        settings_layout.addWidget(QtWidgets.QLabel("FFT Size:"), row, 2)
        self.fft_size_combo = QtWidgets.QComboBox()
        self.fft_size_combo.addItems(['512', '1024', '2048', '4096'])
        self.fft_size_combo.setCurrentText('1024')
        settings_layout.addWidget(self.fft_size_combo, row, 3)

        row += 1
        # Color Scheme settings
        settings_layout.addWidget(QtWidgets.QLabel("Color Scheme:"), row, 0)
        self.color_scheme_combo = QtWidgets.QComboBox()
        self.color_scheme_combo.addItems(['Viridis', 'Plasma', 'Inferno', 'Magma'])
        self.color_scheme_combo.currentTextChanged.connect(self.update_color_scheme)
        settings_layout.addWidget(self.color_scheme_combo, row, 1)

        row += 1
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Detection")
        self.start_button.clicked.connect(self.toggle_detection)

        self.peak_hold_button = QtWidgets.QPushButton("Enable Peak Hold")
        self.peak_hold_button.setCheckable(True)
        self.peak_hold_button.clicked.connect(self.toggle_peak_hold)

        self.reset_button = QtWidgets.QPushButton("Reset SDR")
        self.reset_button.clicked.connect(self.reset_sdr)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.peak_hold_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()

        settings_layout.addLayout(button_layout, row, 0, 1, 4)

        row += 1
        # Status label
        self.status_label = QtWidgets.QLabel("Stopped")
        self.status_label.setStyleSheet("color: #EBCB8B;")
        settings_layout.addWidget(self.status_label, row, 0, 1, 4)

        h_layout.addWidget(settings_panel)

        # LoRa Detection Info Panel
        info_panel = QtWidgets.QWidget()
        info_layout = QtWidgets.QGridLayout(info_panel)
        info_panel.setStyleSheet("""
            QWidget {
                background-color: #3B4252;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: #ECEFF4;
            }
            QLabel#title {
                font-weight: bold;
                color: #88C0D0;
                font-size: 14px;
            }
            QLabel#value {
                color: #A3BE8C;
                font-family: monospace;
            }
            QFrame {
                border: 1px solid #4C566A;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        title = QtWidgets.QLabel("LoRa Signal Detection")
        title.setObjectName("title")
        info_layout.addWidget(title, 0, 0, 1, 2)

        labels = [
            ("Detected Frequency:", "freq_label", "- MHz"),
            ("Signal Strength:", "power_label", "- dBm"),
            ("Bandwidth:", "bw_label", "- kHz"),
            ("Spreading Factor:", "sf_label", "SF-"),
            ("SNR:", "snr_label", "- dB"),
            ("Last Detection:", "time_label", "-"),
            ("Detection Count:", "count_label", "0"),
            ("Channel Activity:", "activity_label", "No Activity")
        ]

        for i, (text, name, initial) in enumerate(labels, start=1):
            label = QtWidgets.QLabel(text)
            value = QtWidgets.QLabel(initial)
            value.setObjectName("value")
            setattr(self, name, value)
            info_layout.addWidget(label, i, 0)
            info_layout.addWidget(value, i, 1)

        h_layout.addWidget(info_panel)
        layout.addLayout(h_layout)

        # Add detection table
        table_label = QtWidgets.QLabel("Detected LoRa Signals")
        table_label.setStyleSheet("color: #ECEFF4; font-weight: bold;")
        layout.addWidget(table_label)
        self.table_widget = QtWidgets.QTableWidget()
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels([
            "Frequency (MHz)", "Power (dBm)", "SNR (dB)", 
            "Spreading Factor (SF)", "Bandwidth (kHz)", "Detection Time"
        ])
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table_widget)

    def update_gain_combo(self):
        try:
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
        except Exception as e:
            logging.error(f"Error updating gain combo: {str(e)}")

    def handle_preset_change(self, preset):
        if preset != "Custom":
            try:
                freq = float(preset.split()[0])
                self.freq_input.setText(str(freq))
            except ValueError:
                logging.error(f"Invalid preset frequency format: {preset}")

    def update_detection_info(self, detected_freq, power, snr, sf, bw, detection_time):
        try:
            self.freq_label.setText(f"{detected_freq:.3f} MHz")
            self.power_label.setText(f"{power:.1f} dBm")
            self.bw_label.setText(f"{bw/1000:.1f} kHz")
            self.sf_label.setText(f"SF-{sf}")
            self.snr_label.setText(f"{snr:.1f} dB")
            self.time_label.setText(detection_time.strftime("%H:%M:%S.%f")[:-4])
            self.detection_count += 1
            self.count_label.setText(str(self.detection_count))
            self.activity_label.setText("Signal Detected!")
            self.activity_label.setStyleSheet("color: #A3BE8C;")
            
            # Add detection to the table
            row_position = self.table_widget.rowCount()
            self.table_widget.insertRow(row_position)
            self.table_widget.setItem(row_position, 0, QtWidgets.QTableWidgetItem(f"{detected_freq:.3f}"))
            self.table_widget.setItem(row_position, 1, QtWidgets.QTableWidgetItem(f"{power:.1f}"))
            self.table_widget.setItem(row_position, 2, QtWidgets.QTableWidgetItem(f"{snr:.1f}"))
            self.table_widget.setItem(row_position, 3, QtWidgets.QTableWidgetItem(f"{sf}"))
            self.table_widget.setItem(row_position, 4, QtWidgets.QTableWidgetItem(f"{bw/1000:.1f}"))
            self.table_widget.setItem(row_position, 5, QtWidgets.QTableWidgetItem(detection_time.strftime("%H:%M:%S")))
            
            # Scroll to the newest entry
            self.table_widget.scrollToBottom()
        except Exception as e:
            logging.error(f"Error updating detection info: {str(e)}")

    def clear_detection_info(self):
        self.activity_label.setText("No Activity")
        self.activity_label.setStyleSheet("color: #EBCB8B;")

    def toggle_detection(self):
        try:
            if self.running:
                self.stop_detection()
            else:
                self.start_detection()
        except Exception as e:
            logging.error(f"Error in toggle_detection: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    def start_detection(self):
        try:
            if self.sdr_controller.is_active:
                self.status_label.setText("SDR is being used by another component.")
                self.status_label.setStyleSheet("color: #BF616A;")
                logging.warning("SDR is active. Cannot start detection.")
                return

            # Validate and set SDR parameters
            center_freq = float(self.freq_input.text()) * 1e6
            sample_rate = float(self.sample_rate_input.text()) * 1e6
            gain = self.gain_combo.currentText()
            fft_size = int(self.fft_size_combo.currentText())

            # Input validation
            if not (24e6 <= center_freq <= 1.766e9):
                raise ValueError("Center frequency must be between 24 MHz and 1.766 GHz")
            if not (1.0e6 <= sample_rate <= 3.2e6):
                raise ValueError("Sample rate must be between 1.0 MHz and 3.2 MHz")

            # Configure SDR
            self.sdr_controller.center_freq = center_freq
            self.sdr_controller.sample_rate = sample_rate
            self.sdr_controller.fft_size = fft_size

            # Set gain
            if gain.lower() == 'auto':
                self.sdr_controller.gain = None
            else:
                self.sdr_controller.gain = float(gain)

            # Setup SDR
            if self.sdr_controller.setup():
                self.timer.start(200)  # Update every 200ms
                self.start_button.setText("Stop Detection")
                self.status_label.setText("Detecting")
                self.status_label.setStyleSheet("color: #A3BE8C;")
                self.detection_count = 0  # Reset detection count
                self.table_widget.setRowCount(0)  # Clear the table
                self.running = True
                logging.info("Started LoRa detection")
            else:
                self.status_label.setText("Failed to set up SDR")
                self.status_label.setStyleSheet("color: #BF616A;")
                logging.error("Failed to set up SDR")

        except ValueError as ve:
            self.status_label.setText(str(ve))
            self.status_label.setStyleSheet("color: #BF616A;")
            logging.error(f"Validation error: {str(ve)}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")
            logging.error(f"Error starting detection: {str(e)}")

    def stop_detection(self):
        try:
            # Stop the timer first
            self.timer.stop()

            # Stop all detection threads
            self.thread_manager.stop_all()

            # Clean up SDR
            self.sdr_controller.close()

            # Update UI
            self.start_button.setText("Start Detection")
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("color: #EBCB8B;")
            self.clear_detection_info()
            self.running = False
            logging.info("Stopped LoRa detection")

        except Exception as e:
            logging.error(f"Error in stop_detection: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    def reset_sdr(self):
        try:
            self.stop_detection()
            if self.sdr_controller.reset():
                self.status_label.setText("SDR Reset Successfully")
                self.status_label.setStyleSheet("color: #A3BE8C;")
                self.update_gain_combo()
                logging.info("SDR reset successfully")
            else:
                self.status_label.setText("Failed to Reset SDR")
                self.status_label.setStyleSheet("color: #BF616A;")
                logging.error("Failed to reset SDR")
        except Exception as e:
            logging.error(f"Error in reset_sdr: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    def update_plot(self):
        try:
            if not self.running:
                return

            samples = self.sdr_controller.read_samples()
            if samples is None:
                raise ValueError("Failed to read samples from SDR")

            freq_range, power_spectrum_dbm = self.sdr_controller.compute_power_spectrum(samples)

            if freq_range is not None and power_spectrum_dbm is not None:
                self.spectrum_widget.update(freq_range, power_spectrum_dbm)
                self.waterfall_widget.update(power_spectrum_dbm)

                # Start detection thread
                self.start_detection_thread(samples)
            else:
                raise ValueError("Invalid spectrum data received")

        except Exception as e:
            logging.error(f"Error updating plot: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")
            self.stop_detection()

    def start_detection_thread(self, samples):
        try:
            # Get detection parameters
            sf_selection = self.sf_combo.currentText()
            if sf_selection == 'All':
                sf_values = [7, 8, 9, 10, 11, 12]
            else:
                try:
                    sf_values = [int(sf_selection)]
                except ValueError:
                    sf_values = []
                    logging.error(f"Invalid SF selection: {sf_selection}")

            bw_selection = self.bw_combo.currentText()
            if bw_selection == 'All':
                bw_values = [125e3, 250e3, 500e3]
            else:
                try:
                    bw_values = [float(bw_selection) * 1e3]  # Convert kHz to Hz
                except ValueError:
                    bw_values = []
                    logging.error(f"Invalid BW selection: {bw_selection}")

            try:
                threshold_db = float(self.threshold_input.text())
            except ValueError:
                threshold_db = 3.0  # Default value
                logging.error(f"Invalid threshold value: {self.threshold_input.text()}")

            # Create worker and thread
            worker = DetectionWorker(
                samples,
                self.sdr_controller.sample_rate,
                sf_values,
                bw_values,
                threshold_db,
                self.sdr_controller.center_freq
            )

            thread = QThread()
            worker.moveToThread(thread)
            
            # Connect signals
            thread.started.connect(worker.process)
            worker.detection_finished.connect(self.on_detection_finished)
            worker.detection_finished.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            # Add to thread manager and start
            self.thread_manager.add_thread(thread)
            thread.start()

        except Exception as e:
            logging.error(f"Error starting detection thread: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    @QtCore.pyqtSlot(bool, float, float, float, int, float)
    def on_detection_finished(self, detected, detected_freq, power_dbm, snr, sf, bw):
        if detected and self.running:
            self.update_detection_info(
                detected_freq,
                power_dbm,
                snr,
                sf,
                bw,
                datetime.now()
            )
            self.status_label.setText("LoRa Signal Detected")
            self.status_label.setStyleSheet("color: #A3BE8C;")
        else:
            self.clear_detection_info()

    def toggle_peak_hold(self):
        try:
            self.spectrum_widget.set_peak_hold(self.peak_hold_button.isChecked())
        except Exception as e:
            logging.error(f"Error toggling peak hold: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    def update_color_scheme(self, scheme):
        try:
            self.waterfall_widget.set_color_scheme(scheme)
            logging.info(f"Updated color scheme to {scheme}")
        except Exception as e:
            logging.error(f"Error updating color scheme: {str(e)}")
            self.status_label.setText(f"Error updating color scheme: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    def closeEvent(self, event):
        try:
            self.stop_detection()
            event.accept()
        except Exception as e:
            logging.error(f"Error in closeEvent: {str(e)}")
            event.accept()  # Accept the close event even if there's an error