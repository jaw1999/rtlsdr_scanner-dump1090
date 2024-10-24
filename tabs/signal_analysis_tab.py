# signal_analysis_tab.py

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy import signal
from collections import deque

class DataProcessingThread(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(object)
    stats_ready = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    progress_updated = QtCore.pyqtSignal(int)

    def __init__(self, sdr_controller, demod_type):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.demod_type = demod_type
        self.is_running = True

    def run(self):
        progress = 0
        while self.is_running:
            try:
                samples = self.sdr_controller.read_samples()
                if samples is None:
                    continue

                i_data = np.real(samples)
                q_data = np.imag(samples)

                freq_range, power_spectrum_dbm = self.sdr_controller.compute_power_spectrum(samples)
                if freq_range is None or power_spectrum_dbm is None:
                    continue

                demodulated = self.demodulate(samples, self.demod_type)

                data = (i_data, q_data, freq_range, power_spectrum_dbm, demodulated)
                self.data_ready.emit(data)

                self.update_signal_statistics(samples, freq_range, power_spectrum_dbm)

                # Update progress bar
                progress = (progress + 1) % 101
                self.progress_updated.emit(progress)

                self.msleep(10)
            except Exception as e:
                self.error_occurred.emit(f"Error in data processing: {str(e)}")
                break

    def demodulate(self, samples, demod_type):
        if demod_type == "AM":
            return np.abs(samples)
        elif demod_type == "FM":
            return np.unwrap(np.angle(samples[1:] * np.conj(samples[:-1])))
        elif demod_type == "USB":
            return np.real(signal.hilbert(np.real(samples)))
        elif demod_type == "LSB":
            return np.imag(signal.hilbert(np.imag(samples)))
        elif demod_type == "CW":
            return np.abs(signal.hilbert(np.real(samples)))
        else:
            return np.real(samples)

    def update_signal_statistics(self, samples, freq_range, power_spectrum_dbm):
        try:
            mean = np.mean(samples)
            std_dev = np.std(samples)
            max_amp = np.max(np.abs(samples))
            total_power = np.sum(np.abs(samples) ** 2)
            avg_power = total_power / len(samples)
            peak_freq = freq_range[np.argmax(power_spectrum_dbm)]
            bandwidth = self.estimate_bandwidth(freq_range, power_spectrum_dbm)
            noise_floor = np.median(power_spectrum_dbm)
            signal_power = np.max(power_spectrum_dbm)
            snr = signal_power - noise_floor  # SNR in dB

            # Additional statistics
            crest_factor = max_amp / avg_power
            rms = np.sqrt(np.mean(np.abs(samples) ** 2))
            dynamic_range = signal_power - noise_floor

            stats_text = f"""
            <html>
            <head/>
            <body>
            <h2 style="color: #00CED1;">Signal Statistics</h2>
            <table style="color: #FFFFFF; font-family: Arial; font-size: 12px;">
                <tr><td><b>Mean Amplitude:</b></td><td>{mean.real:.4f} + {mean.imag:.4f}j</td></tr>
                <tr><td><b>Standard Deviation:</b></td><td>{std_dev:.4f}</td></tr>
                <tr><td><b>Max Amplitude:</b></td><td>{max_amp:.4f}</td></tr>
                <tr><td><b>RMS Amplitude:</b></td><td>{rms:.4f}</td></tr>
                <tr><td><b>Crest Factor:</b></td><td>{crest_factor:.4f}</td></tr>
                <tr><td><b>Total Power:</b></td><td>{total_power:.4f}</td></tr>
                <tr><td><b>Average Power:</b></td><td>{avg_power:.4f}</td></tr>
                <tr><td><b>Peak Frequency:</b></td><td>{peak_freq:.4f} MHz</td></tr>
                <tr><td><b>Estimated Bandwidth:</b></td><td>{bandwidth * 1e-3:.2f} kHz</td></tr>
                <tr><td><b>Estimated SNR:</b></td><td>{snr:.2f} dB</td></tr>
                <tr><td><b>Dynamic Range:</b></td><td>{dynamic_range:.2f} dB</td></tr>
            </table>
            </body>
            </html>
            """
            self.stats_ready.emit(stats_text)
        except Exception as e:
            self.error_occurred.emit(f"Error updating statistics: {str(e)}")

    def estimate_bandwidth(self, freq_range, power_spectrum_dbm):
        peak_power = np.max(power_spectrum_dbm)
        threshold = peak_power - 3  # 3 dB bandwidth
        above_threshold = power_spectrum_dbm > threshold
        if np.any(above_threshold):
            bandwidth = freq_range[above_threshold][-1] - freq_range[above_threshold][0]
        else:
            bandwidth = 0
        return bandwidth

    def stop(self):
        self.is_running = False
        self.wait()

class SignalAnalysisTab(QtWidgets.QWidget):
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.is_processing = False
        self.processing_thread = None
        self.waterfall_data = deque(maxlen=500)  # Increased buffer size
        self.fft_size = 1024
        self.spectrogram_data = np.zeros((500, self.fft_size))

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Signal Analysis")
        self.setStyleSheet(self.get_stylesheet())

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Control Panel
        control_panel = self.create_control_panel()
        main_layout.addLayout(control_panel)

        # Analysis Tabs
        self.analysis_tabs = self.create_analysis_tabs()
        main_layout.addWidget(self.analysis_tabs)

        # Progress Bar
        self.progress_bar = self.create_progress_bar()
        main_layout.addWidget(self.progress_bar)

    def create_control_panel(self):
        control_panel = QtWidgets.QHBoxLayout()
        control_panel.setSpacing(15)

        # Start/Stop Button
        self.start_button = QtWidgets.QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.toggle_analysis)
        self.start_button.setToolTip("Start or stop the signal analysis.")
        control_panel.addWidget(self.start_button)

        # Frequency Input
        freq_layout = QtWidgets.QVBoxLayout()
        freq_label = QtWidgets.QLabel("Center Frequency (MHz):")
        self.freq_input = QtWidgets.QDoubleSpinBox()
        self.freq_input.setRange(24, 1766)
        self.freq_input.setDecimals(3)
        self.freq_input.setSuffix(" MHz")
        self.freq_input.setValue(100.0)
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)
        control_panel.addLayout(freq_layout)

        # Sample Rate Input
        rate_layout = QtWidgets.QVBoxLayout()
        rate_label = QtWidgets.QLabel("Sample Rate (MHz):")
        self.sample_rate_input = QtWidgets.QDoubleSpinBox()
        self.sample_rate_input.setRange(0.9, 3.2)
        self.sample_rate_input.setDecimals(1)
        self.sample_rate_input.setSuffix(" MHz")
        self.sample_rate_input.setValue(2.4)
        rate_layout.addWidget(rate_label)
        rate_layout.addWidget(self.sample_rate_input)
        control_panel.addLayout(rate_layout)

        # Demodulation ComboBox
        demod_layout = QtWidgets.QVBoxLayout()
        demod_label = QtWidgets.QLabel("Demodulation Type:")
        self.demod_combo = QtWidgets.QComboBox()
        self.demod_combo.addItems(["None", "AM", "FM", "USB", "LSB", "CW"])
        demod_layout.addWidget(demod_label)
        demod_layout.addWidget(self.demod_combo)
        control_panel.addLayout(demod_layout)

        # FFT Size ComboBox
        fft_layout = QtWidgets.QVBoxLayout()
        fft_label = QtWidgets.QLabel("FFT Size:")
        self.fft_size_combo = QtWidgets.QComboBox()
        self.fft_size_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.fft_size_combo.setCurrentText("1024")
        self.fft_size_combo.currentTextChanged.connect(self.update_fft_size)
        fft_layout.addWidget(fft_label)
        fft_layout.addWidget(self.fft_size_combo)
        control_panel.addLayout(fft_layout)

        # Colormap Selection
        cmap_layout = QtWidgets.QVBoxLayout()
        cmap_label = QtWidgets.QLabel("Colormap:")
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis"])
        self.cmap_combo.currentTextChanged.connect(self.update_colormap)
        cmap_layout.addWidget(cmap_label)
        cmap_layout.addWidget(self.cmap_combo)
        control_panel.addLayout(cmap_layout)

        return control_panel

    def create_analysis_tabs(self):
        tabs = QtWidgets.QTabWidget()
        tabs.setTabPosition(QtWidgets.QTabWidget.North)

        # IQ Plot
        self.iq_plot = pg.PlotWidget(title="IQ Plot")
        self.iq_plot.setLabel('left', "Q")
        self.iq_plot.setLabel('bottom', "I")
        self.iq_scatter = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 255, 120))
        self.iq_plot.addItem(self.iq_scatter)
        tabs.addTab(self.iq_plot, "IQ Plot")

        # Spectrum Plot
        self.spectrum_plot = pg.PlotWidget(title="Spectrum")
        self.spectrum_plot.setLabel('left', "Power (dBm)")
        self.spectrum_plot.setLabel('bottom', "Frequency (MHz)")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(color='c', width=2))
        tabs.addTab(self.spectrum_plot, "Spectrum")

        # Waterfall Plot
        self.waterfall_plot = pg.PlotWidget(title="Waterfall")
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        self.waterfall_plot.setLabel('left', "Time")
        self.waterfall_plot.setLabel('bottom', "Frequency (MHz)")
        self.waterfall_image.setLookupTable(self.get_colormap(self.cmap_combo.currentText()))
        self.waterfall_image.setAutoDownsample(True)
        tabs.addTab(self.waterfall_plot, "Waterfall")

        # Spectrogram Plot
        self.spectrogram_plot = pg.PlotWidget(title="Spectrogram")
        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.spectrogram_image)
        self.spectrogram_plot.setLabel('left', "Frequency (MHz)")
        self.spectrogram_plot.setLabel('bottom', "Time")
        self.spectrogram_image.setLookupTable(self.get_colormap(self.cmap_combo.currentText()))
        self.spectrogram_image.setAutoDownsample(True)
        tabs.addTab(self.spectrogram_plot, "Spectrogram")

        # Demodulated Signal Plot
        self.demod_plot = pg.PlotWidget(title="Demodulated Signal")
        self.demod_plot.setLabel('left', "Amplitude")
        self.demod_plot.setLabel('bottom', "Time")
        self.demod_curve = self.demod_plot.plot(pen=pg.mkPen(color='y', width=2))
        tabs.addTab(self.demod_plot, "Demodulated Signal")

        # Constellation Diagram
        self.constellation_plot = pg.PlotWidget(title="Constellation Diagram")
        self.constellation_plot.setLabel('left', "Q")
        self.constellation_plot.setLabel('bottom', "I")
        self.constellation_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120))
        self.constellation_plot.addItem(self.constellation_scatter)
        tabs.addTab(self.constellation_plot, "Constellation")

        # Signal Statistics
        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #2C2C2C;
                color: #FFFFFF;
                font-family: Arial;
                font-size: 12px;
                border: none;
            }
        """)
        tabs.addTab(self.stats_text, "Signal Statistics")

        return tabs

    def create_progress_bar(self):
        progress_bar = QtWidgets.QProgressBar(self)
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3A3A3A;
                border-radius: 5px;
                text-align: center;
                background-color: #1E1E1E;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        return progress_bar

    def toggle_analysis(self):
        if self.is_processing:
            self.stop_analysis()
        else:
            self.start_analysis()

    def start_analysis(self):
        center_freq = self.freq_input.value() * 1e6
        sample_rate = self.sample_rate_input.value() * 1e6

        if not self.validate_input(center_freq, sample_rate):
            return

        try:
            self.sdr_controller.center_freq = center_freq
            self.sdr_controller.sample_rate = sample_rate

            if not self.sdr_controller.setup():
                raise Exception("Failed to set up SDR")

            self.is_processing = True
            self.start_button.setText("Stop Analysis")
            self.start_button.setStyleSheet("background-color: #F44336; color: white;")

            demod_type = self.demod_combo.currentText()
            self.processing_thread = DataProcessingThread(self.sdr_controller, demod_type)

            # Connect signals
            self.processing_thread.data_ready.connect(self.update_plots_gui)
            self.processing_thread.stats_ready.connect(self.update_stats_gui)
            self.processing_thread.error_occurred.connect(self.show_error_message)
            self.processing_thread.progress_updated.connect(self.progress_bar.setValue)

            self.processing_thread.start()
        except Exception as e:
            self.error_signal.emit(f"Failed to start analysis: {str(e)}")

    def stop_analysis(self):
        if self.is_processing and self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread = None
            self.is_processing = False
            self.start_button.setText("Start Analysis")
            self.start_button.setStyleSheet("")
            self.sdr_controller.close()

    def update_plots_gui(self, data):
        i_data, q_data, freq_range, power_spectrum_dbm, demodulated = data

        # Update IQ Plot
        self.iq_scatter.setData(i_data[:1000], q_data[:1000])

        # Update Spectrum Plot
        freq_range_mhz = freq_range / 1e6
        self.spectrum_curve.setData(freq_range_mhz, power_spectrum_dbm)

        # Update Waterfall Plot
        power_spectrum_dbm_clipped = np.clip(power_spectrum_dbm, -120, 0)
        self.waterfall_data.append(power_spectrum_dbm_clipped)
        waterfall_array = np.array(self.waterfall_data)
        self.waterfall_image.setImage(waterfall_array.T, autoLevels=False)
        self.waterfall_image.setLevels([power_spectrum_dbm_clipped.min(), power_spectrum_dbm_clipped.max()])

        # Update Spectrogram Plot
        self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=0)
        self.spectrogram_data[-1, :] = power_spectrum_dbm_clipped
        self.spectrogram_image.setImage(self.spectrogram_data.T, autoLevels=False)
        self.spectrogram_image.setLevels([power_spectrum_dbm_clipped.min(), power_spectrum_dbm_clipped.max()])

        # Update Demodulated Signal Plot
        self.demod_curve.setData(demodulated[:1000])

        # Update Constellation Diagram
        self.constellation_scatter.setData(i_data[:1000], q_data[:1000])

    def update_stats_gui(self, stats_text):
        self.stats_text.setHtml(stats_text.strip())

    def validate_input(self, center_freq, sample_rate):
        if not (24e6 <= center_freq <= 1766e6):
            self.error_signal.emit("Center frequency must be between 24 MHz and 1766 MHz.")
            return False
        if not (0.9e6 <= sample_rate <= 3.2e6):
            self.error_signal.emit("Sample rate must be between 0.9 MHz and 3.2 MHz.")
            return False
        return True

    def update_fft_size(self, size):
        self.fft_size = int(size)
        self.sdr_controller.fft_size = self.fft_size
        self.spectrogram_data = np.zeros((500, self.fft_size))

    def update_colormap(self, cmap_name):
        # Update colormap for waterfall and spectrogram
        colormap = self.get_colormap(cmap_name)
        self.waterfall_image.setLookupTable(colormap)
        self.spectrogram_image.setLookupTable(colormap)

    def show_error_message(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", message)

    def get_stylesheet(self):
        return """
        QWidget {
            background-color: #1E1E1E;
            color: #FFFFFF;
            font-family: Arial;
            font-size: 12px;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            padding: 8px;
            border: none;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QDoubleSpinBox, QComboBox, QLabel {
            padding: 5px;
        }
        QTabWidget::pane {
            border: 1px solid #3A3A3A;
        }
        QTabBar::tab {
            background-color: #2C2C2C;
            color: #FFFFFF;
            padding: 8px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #3A3A3A;
        }
        QTextEdit {
            background-color: #2C2C2C;
            color: #FFFFFF;
            border: none;
        }
        """

    def get_colormap(self, cmap_name='viridis'):
        """
        Returns a lookup table for the specified colormap.
        Supported colormap names: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        """
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(cmap_name)
            colormap = cmap(np.linspace(0.0, 1.0, 256))
            colormap = (colormap[:, :3] * 255).astype(np.uint8)
            return colormap
        except ImportError:
            # Fallback to a default colormap if matplotlib is not available
            colors = [
                (0, 0, 0),
                (0, 0, 128),
                (0, 0, 255),
                (0, 255, 255),
                (255, 255, 0),
                (255, 128, 0),
                (255, 0, 0),
                (255, 255, 255)
            ]
            positions = np.linspace(0.0, 1.0, len(colors))
            colormap = pg.ColorMap(positions, colors)
            return colormap.getLookupTable()

    def closeEvent(self, event):
        self.stop_analysis()
        super().closeEvent(event)
