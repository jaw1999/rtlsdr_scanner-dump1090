from PyQt5 import QtWidgets, QtCore
import numpy as np
import logging
from widgets.spectrum_lora_widget import SpectrumLoRaWidget
from widgets.waterfall_lora_widget import WaterfallLoRaWidget
from datetime import datetime
from scipy.signal import fftconvolve
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QTimer
from typing import List, Optional, Deque
from collections import deque
import time

# Constants
MAX_TABLE_ROWS = 1000
MAX_WATERFALL_HISTORY = 50
UPDATE_INTERVAL_MS = 50
SAMPLE_BUFFER_SIZE = 4096
MAX_FRAME_SKIP = 2
MIN_UPDATE_INTERVAL = 0.033
POWER_CALIBRATION_OFFSET = -30  # DBM calibration offset

class RingBuffer:
    """Efficient ring buffer implementation for samples"""
    def __init__(self, maxlen):
        self.buffer = np.zeros(maxlen, dtype=np.complex64)
        self.maxlen = maxlen
        self.index = 0
        self.is_filled = False

    def extend(self, data):
        data_len = len(data)
        if data_len >= self.maxlen:
            self.buffer = data[-self.maxlen:]
            self.index = 0
            self.is_filled = True
            return

        space_left = self.maxlen - self.index
        if data_len > space_left:
            # Fill to the end and wrap around
            self.buffer[self.index:] = data[:space_left]
            self.buffer[:data_len-space_left] = data[space_left:]
            self.index = data_len - space_left
        else:
            # Just add the data
            self.buffer[self.index:self.index+data_len] = data
            self.index = (self.index + data_len) % self.maxlen

        if self.index == 0:
            self.is_filled = True

    def get_all(self):
        if not self.is_filled:
            return self.buffer[:self.index]
        if self.index == 0:
            return self.buffer
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))

    def clear(self):
        self.buffer = np.zeros(self.maxlen, dtype=np.complex64)
        self.index = 0
        self.is_filled = False

class PerformanceMonitor:
    """Monitors and manages performance metrics"""
    def __init__(self):
        self.last_update_time = time.time()
        self.frame_times = deque(maxlen=30)
        self.skip_counter = 0

    def should_process_frame(self):
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed < MIN_UPDATE_INTERVAL:
            self.skip_counter += 1
            if self.skip_counter <= MAX_FRAME_SKIP:
                return False
        
        self.frame_times.append(elapsed)
        self.last_update_time = current_time
        self.skip_counter = 0
        return True

    def get_fps(self):
        if not self.frame_times:
            return 0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

class ThreadManager:
    """Enhanced thread lifecycle manager"""
    def __init__(self):
        self.active_threads: List[QThread] = []
        self._cleanup_lock = QtCore.QMutex()
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_finished_threads)
        self._cleanup_timer.start(1000)

    def add_thread(self, thread: QThread):
        with QtCore.QMutexLocker(self._cleanup_lock):
            thread.finished.connect(lambda: self.remove_thread(thread))
            self.active_threads.append(thread)

    def remove_thread(self, thread: QThread):
        with QtCore.QMutexLocker(self._cleanup_lock):
            if thread in self.active_threads:
                self.active_threads.remove(thread)

    def _cleanup_finished_threads(self):
        with QtCore.QMutexLocker(self._cleanup_lock):
            self.active_threads = [t for t in self.active_threads if t.isRunning()]

    def stop_all(self):
        with QtCore.QMutexLocker(self._cleanup_lock):
            for thread in self.active_threads[:]:
                try:
                    if thread.isRunning():
                        thread.quit()
                        if not thread.wait(1000):
                            thread.terminate()
                except RuntimeError:
                    continue
            self.active_threads.clear()

class DetectionWorker(QObject):
    """Optimized detection worker with improved LoRa-specific detection"""
    detection_finished = pyqtSignal(bool, float, float, float, int, float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, samples, sample_rate, sf_values, bw_values, threshold_db, center_freq):
        super().__init__()
        self.samples = samples
        self.sample_rate = sample_rate
        self.sf_values = sf_values
        self.bw_values = bw_values
        self.threshold_db = threshold_db
        self.center_freq = center_freq
        self._is_running = True
        self._cached_chirps = {}
        
        # LoRa-specific detection parameters
        self.min_snr_threshold = 6.0
        self.chirp_correlation_threshold = 0.7
        self.consecutive_detections_required = 2
        self.detection_history = []
        self.max_history_length = 3

    def stop(self):
        """Stop the worker"""
        self._is_running = False

    def _generate_chirp(self, sf, bw, num_samples):
        """Cached chirp generation"""
        key = (sf, bw, num_samples)
        if key not in self._cached_chirps:
            t = np.arange(num_samples) / self.sample_rate
            k = bw / (2 ** sf)
            self._cached_chirps[key] = np.exp(1j * 2 * np.pi * (0.5 * k * t ** 2))
        return self._cached_chirps[key]

    def validate_samples(self):
        """Validate input samples"""
        if self.samples is None or len(self.samples) == 0:
            raise ValueError("Invalid samples received")
        if np.isnan(self.samples).any() or np.isinf(self.samples).any():
            raise ValueError("Invalid sample values detected")

    @QtCore.pyqtSlot()
    def process(self):
        """Process the samples and detect LoRa signals"""
        try:
            if self._is_running:
                self.validate_samples()
                detected, freq, power, snr, sf, bw = self.detect_lora_signal()
                if self._is_running:
                    self.detection_finished.emit(detected, freq, power, snr, sf, bw)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _validate_chirp_characteristics(self, correlation, magnitude, sf, bw):
        """Validate if the detected signal matches LoRa chirp characteristics"""
        try:
            # 1. Check chirp duration
            expected_symbol_duration = (2**sf) / bw
            actual_duration = len(correlation) / self.sample_rate
            if not (0.8 * expected_symbol_duration <= actual_duration <= 1.2 * expected_symbol_duration):
                return False, 0.0

            # 2. Check for linear frequency sweep
            analytic_signal = correlation
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            phase_diff = np.diff(instantaneous_phase)
            phase_diff_2nd = np.diff(phase_diff)
            phase_linearity = np.std(phase_diff_2nd) / np.mean(np.abs(phase_diff_2nd))
            
            if phase_linearity > 0.3:
                return False, 0.0

            # 3. Check magnitude distribution
            sorted_magnitude = np.sort(magnitude)
            background_level = np.median(sorted_magnitude[:-10])
            peak_to_background = np.max(magnitude) / (background_level + 1e-9)
            
            if peak_to_background < 3.0:
                return False, 0.0

            # 4. Compute correlation quality score
            correlation_quality = peak_to_background * (1.0 - phase_linearity)
            
            return True, correlation_quality

        except Exception as e:
            logging.error(f"Error in chirp validation: {str(e)}")
            return False, 0.0

    def _check_detection_consistency(self, detection_info):
        """Check if detection is consistent with recent history"""
        if not self.detection_history:
            self.detection_history.append(detection_info)
            return False

        self.detection_history.append(detection_info)
        if len(self.detection_history) > self.max_history_length:
            self.detection_history.pop(0)

        if len(self.detection_history) >= self.consecutive_detections_required:
            recent_detections = self.detection_history[-self.consecutive_detections_required:]
            
            freq_differences = [abs(recent_detections[i]['freq'] - recent_detections[i-1]['freq']) 
                              for i in range(1, len(recent_detections))]
            
            sf_consistent = len(set(d['sf'] for d in recent_detections)) == 1
            bw_consistent = len(set(d['bw'] for d in recent_detections)) == 1
            
            max_freq_difference = max(freq_differences) if freq_differences else float('inf')
            freq_consistent = max_freq_difference < (recent_detections[0]['bw'] * 0.01)
            
            return freq_consistent and sf_consistent and bw_consistent

        return False

    def detect_lora_signal(self):
        """Detect LoRa signals in the samples"""
        try:
            detected_signals = []
            num_samples = len(self.samples)

            for sf in self.sf_values:
                if not self._is_running:
                    break
                    
                for bw in self.bw_values:
                    if not self._is_running:
                        break
                        
                    if self.sample_rate < bw:
                        continue

                    ref_chirp = self._generate_chirp(sf, bw, num_samples)
                    correlation = fftconvolve(self.samples, ref_chirp[::-1].conj(), mode='same')
                    magnitude = np.abs(correlation)
                    
                    peak_magnitude = np.max(magnitude)
                    noise_floor = np.median(magnitude)
                    snr = 20 * np.log10(peak_magnitude / (noise_floor + 1e-9))
                    threshold = noise_floor * 10 ** (self.threshold_db / 20)

                    if peak_magnitude > threshold and snr > self.min_snr_threshold:
                        is_valid_chirp, correlation_quality = self._validate_chirp_characteristics(
                            correlation, magnitude, sf, bw)
                        
                        if is_valid_chirp and correlation_quality > self.chirp_correlation_threshold:
                            detected_signals.append({
                                'sf': sf,
                                'bw': bw,
                                'snr': snr,
                                'magnitude': peak_magnitude,
                                'peak_index': np.argmax(magnitude),
                                'correlation_quality': correlation_quality,
                                'freq': (self.center_freq + 
                                       (np.argmax(magnitude) - num_samples // 2) * 
                                       (self.sample_rate / num_samples)) / 1e6
                            })

            if detected_signals and self._is_running:
                best_signal = max(detected_signals, key=lambda x: x['correlation_quality'])
                
                if self._check_detection_consistency(best_signal):
                    freq_resolution = self.sample_rate / num_samples
                    freq_offset = (best_signal['peak_index'] - num_samples // 2) * freq_resolution
                    detected_freq = (self.center_freq + freq_offset) / 1e6
                    power_dbm = 20 * np.log10(best_signal['magnitude']) + POWER_CALIBRATION_OFFSET
                    return True, detected_freq, power_dbm, best_signal['snr'], best_signal['sf'], best_signal['bw']
            
            return False, 0.0, 0.0, 0.0, 0, 0.0

        except Exception as e:
            logging.error(f"Detection error: {str(e)}")
            return False, 0.0, 0.0, 0.0, 0, 0.0

class LoRaTab(QtWidgets.QWidget):
    """Main LoRa detection interface"""
    frequency_changed = pyqtSignal(float)

    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.running = False
        self.worker = None
        self.thread_manager = ThreadManager()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.detection_count = 0
        self.last_detection_time = None
        self.sample_buffer = RingBuffer(SAMPLE_BUFFER_SIZE)
        self.performance_monitor = PerformanceMonitor()
        
        logging.basicConfig(
            filename='lora_tab.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )

        self.init_ui()
        self.frequency_changed.connect(self.handle_frequency_change)

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Create main splitter
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Upper section (display and controls)
        upper_widget = QtWidgets.QWidget()
        upper_layout = QtWidgets.QVBoxLayout(upper_widget)

        # Display container
        display_container = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_container)
        display_container.setStyleSheet("background-color: #2E3440; border-radius: 10px;")

        # Add spectrum widget
        self.spectrum_widget = SpectrumLoRaWidget()
        self.spectrum_widget.plot_widget.setYRange(-120, -20)  # Adjusted DBM range
        display_layout.addWidget(self.spectrum_widget, 3)

        # Add waterfall widget
        self.waterfall_widget = WaterfallLoRaWidget()
        display_layout.addWidget(self.waterfall_widget, 2)

        display_container.setLayout(display_layout)
        upper_layout.addWidget(display_container)

        # Settings and info panels
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
        self.freq_input.textChanged.connect(self.handle_manual_frequency_change)
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
        upper_layout.addLayout(h_layout)

        # Add upper section to main splitter
        upper_widget.setLayout(upper_layout)
        self.main_splitter.addWidget(upper_widget)

        # Detection table section
        table_widget = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_widget)

        # Table controls
        table_header = QtWidgets.QHBoxLayout()
        table_label = QtWidgets.QLabel("Detected LoRa Signals")
        table_label.setStyleSheet("color: #ECEFF4; font-weight: bold;")
        clear_button = QtWidgets.QPushButton("Clear History")
        clear_button.clicked.connect(self.clear_detection_history)
        table_header.addWidget(table_label)
        table_header.addStretch()
        table_header.addWidget(clear_button)
        table_layout.addLayout(table_header)

        # Add detection table
        self.table_widget = QtWidgets.QTableWidget()
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels([
            "Frequency (MHz)", "Power (dBm)", "SNR (dB)", 
            "Spreading Factor (SF)", "Bandwidth (kHz)", "Detection Time"
        ])
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table_layout.addWidget(self.table_widget)

        # Add table section to main splitter
        table_widget.setLayout(table_layout)
        self.main_splitter.addWidget(table_widget)

        # Set initial splitter sizes
        self.main_splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        layout.addWidget(self.main_splitter)

    def handle_manual_frequency_change(self, text):
        """Handle manual frequency input changes"""
        try:
            freq = float(text) * 1e6
            self.frequency_changed.emit(freq)
        except ValueError:
            pass

    def handle_frequency_change(self, new_freq):
        """Handle frequency changes synchronously"""
        try:
            if self.running:
                # Stop current detection
                self.timer.stop()
                
                # Update SDR frequency
                self.sdr_controller.center_freq = new_freq
                
                # Clear displays
                self.spectrum_widget.clear_data()
                self.waterfall_widget.clear_data()
                self.sample_buffer.clear()
                
                # Adjust spectrum widget range
                center_freq_mhz = new_freq / 1e6
                sample_rate_mhz = self.sdr_controller.sample_rate / 1e6
                self.spectrum_widget.plot_widget.setXRange(
                    center_freq_mhz - sample_rate_mhz/2,
                    center_freq_mhz + sample_rate_mhz/2
                )
                
                # Restart detection
                self.timer.start(UPDATE_INTERVAL_MS)
                
                logging.info(f"Frequency changed to {new_freq/1e6:.3f} MHz")
                
        except Exception as e:
            logging.error(f"Error changing frequency: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    def validate_sdr_state(self):
        if not self.sdr_controller:
            raise RuntimeError("SDR controller not initialized")
        if self.sdr_controller.is_active:
            raise RuntimeError("SDR is being used by another component")

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
            raise

    def handle_preset_change(self, preset):
        if preset != "Custom":
            try:
                freq = float(preset.split()[0]) * 1e6
                self.freq_input.setText(str(freq/1e6))
                self.frequency_changed.emit(freq)
            except ValueError:
                logging.error(f"Invalid preset frequency format: {preset}")

    def update_detection_info(self, detected_freq, power, snr, sf, bw, detection_time):
        try:
            # Update labels
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
            
            # Update table with row limit
            row_position = self.table_widget.rowCount()
            if row_position >= MAX_TABLE_ROWS:
                self.table_widget.removeRow(0)
                row_position = MAX_TABLE_ROWS - 1
                
            self.table_widget.insertRow(row_position)
            self.table_widget.setItem(row_position, 0, QtWidgets.QTableWidgetItem(f"{detected_freq:.3f}"))
            self.table_widget.setItem(row_position, 1, QtWidgets.QTableWidgetItem(f"{power:.1f}"))
            self.table_widget.setItem(row_position, 2, QtWidgets.QTableWidgetItem(f"{snr:.1f}"))
            self.table_widget.setItem(row_position, 3, QtWidgets.QTableWidgetItem(f"{sf}"))
            self.table_widget.setItem(row_position, 4, QtWidgets.QTableWidgetItem(f"{bw/1000:.1f}"))
            self.table_widget.setItem(row_position, 5, QtWidgets.QTableWidgetItem(detection_time.strftime("%H:%M:%S")))
            
            self.table_widget.scrollToBottom()
            
        except Exception as e:
            logging.error(f"Error updating detection info: {str(e)}")
            raise

    def clear_detection_history(self):
        """Clear the detection history table"""
        self.table_widget.setRowCount(0)
        self.detection_count = 0
        self.count_label.setText("0")
        self.clear_detection_info()

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

    def setup_sdr(self):
        try:
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

            # Setup SDR and adjust spectrum view
            success = self.sdr_controller.setup()
            if success:
                center_freq_mhz = center_freq / 1e6
                sample_rate_mhz = sample_rate / 1e6
                self.spectrum_widget.plot_widget.setXRange(
                    center_freq_mhz - sample_rate_mhz/2,
                    center_freq_mhz + sample_rate_mhz/2
                )
                self.spectrum_widget.clear_data()
                self.waterfall_widget.clear_data()
                self.sample_buffer.clear()
            return success

        except Exception as e:
            logging.error(f"SDR setup error: {str(e)}")
            raise

    def start_detection(self):
        try:
            self.validate_sdr_state()
            
            if self.setup_sdr():
                self.timer.start(UPDATE_INTERVAL_MS)
                self.start_button.setText("Stop Detection")
                self.status_label.setText("Detecting")
                self.status_label.setStyleSheet("color: #A3BE8C;")
                self.detection_count = 0
                self.table_widget.setRowCount(0)
                self.running = True
                logging.info("Started LoRa detection")
            else:
                raise RuntimeError("Failed to set up SDR")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")
            logging.error(f"Error starting detection: {str(e)}")
            raise

    def stop_detection(self):
        try:
            # Stop the timer first
            self.timer.stop()

            # Stop current worker if exists
            if self.worker:
                self.worker.stop()
                self.worker = None

            # Stop all detection threads
            self.thread_manager.stop_all()

            # Clean up SDR
            self.sdr_controller.close()

            # Clear sample buffer
            self.sample_buffer.clear()

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
            raise

    def reset_sdr(self):
        try:
            self.stop_detection()
            if self.sdr_controller.reset():
                self.status_label.setText("SDR Reset Successfully")
                self.status_label.setStyleSheet("color: #A3BE8C;")
                self.update_gain_combo()
                logging.info("SDR reset successfully")
            else:
                raise RuntimeError("Failed to reset SDR")
        except Exception as e:
            logging.error(f"Error in reset_sdr: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")

    def update_plot(self):
        """Optimized plot update method with power calibration"""
        try:
            if not self.running or not self.performance_monitor.should_process_frame():
                return

            samples = self.sdr_controller.read_samples()
            if samples is None or len(samples) == 0:
                return

            self.sample_buffer.extend(samples)
            
            freq_range, power_spectrum_dbm = self.sdr_controller.compute_power_spectrum(
                self.sample_buffer.get_all()
            )
            
            if freq_range is not None and power_spectrum_dbm is not None:
                # Apply power calibration offset
                power_spectrum_dbm += POWER_CALIBRATION_OFFSET
                
                self.spectrum_widget.update(freq_range, power_spectrum_dbm)
                self.waterfall_widget.update(power_spectrum_dbm)
                self.start_detection_thread(samples)
                
                # Update FPS in status
                fps = self.performance_monitor.get_fps()
                self.status_label.setText(f"Detecting ({fps:.1f} FPS)")

        except Exception as e:
            logging.error(f"Error updating plot: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")
            self.stop_detection()

    def start_detection_thread(self, samples):
        try:
            # Stop existing worker if any
            if self.worker:
                self.worker.stop()
                self.worker = None

            # Get detection parameters
            sf_selection = self.sf_combo.currentText()
            sf_values = [int(sf_selection)] if sf_selection != 'All' else [7, 8, 9, 10, 11, 12]

            bw_selection = self.bw_combo.currentText()
            bw_values = ([float(bw_selection) * 1e3] if bw_selection != 'All' 
                        else [125e3, 250e3, 500e3])

            try:
                threshold_db = float(self.threshold_input.text())
            except ValueError:
                threshold_db = 3.0
                logging.warning(f"Invalid threshold value, using default: {threshold_db}")

            # Create worker and thread
            self.worker = DetectionWorker(
                samples,
                self.sdr_controller.sample_rate,
                sf_values,
                bw_values,
                threshold_db,
                self.sdr_controller.center_freq
            )

            thread = QThread()
            self.worker.moveToThread(thread)
            
            # Connect signals
            thread.started.connect(self.worker.process)
            self.worker.detection_finished.connect(self.on_detection_finished)
            self.worker.error_occurred.connect(self.handle_worker_error)
            thread.finished.connect(thread.deleteLater)

            # Add to thread manager and start
            self.thread_manager.add_thread(thread)
            thread.start()

        except Exception as e:
            logging.error(f"Error starting detection thread: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A;")
            raise

    def handle_worker_error(self, error_msg):
        logging.error(f"Worker error: {error_msg}")
        self.status_label.setText(f"Detection Error: {error_msg}")
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
            self.thread_manager.stop_all()
            event.accept()
        except Exception as e:
            logging.error(f"Error in closeEvent: {str(e)}")
            event.accept()