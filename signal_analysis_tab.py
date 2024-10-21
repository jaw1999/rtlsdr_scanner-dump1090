import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtChart import QChart, QChartView, QScatterSeries
import pyqtgraph as pg
from scipy import signal
import threading
import traceback
from collections import deque

class SignalAnalysisTab(QtWidgets.QWidget):
    update_plots_signal = QtCore.pyqtSignal(object)
    update_stats_signal = QtCore.pyqtSignal(str)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.initUI()
        self.processing_thread = None
        self.is_processing = False
        self.update_plots_signal.connect(self.update_plots_gui)
        self.update_stats_signal.connect(self.update_stats_gui)
        self.error_signal.connect(self.show_error_message)
        self.last_samples = None
        self.waterfall_data = deque(maxlen=256)
        self.waterfall_buffer_size = 10
        self.fft_size = 1024
        self.spectrogram_data = np.zeros((256, self.fft_size))

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Control panel
        control_panel = QtWidgets.QHBoxLayout()
        control_panel.setSpacing(15)

        self.start_button = QtWidgets.QPushButton("Start Analysis")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_button.clicked.connect(self.toggle_analysis)
        control_panel.addWidget(self.start_button)

        self.freq_input = QtWidgets.QDoubleSpinBox()
        self.freq_input.setRange(24, 1766)
        self.freq_input.setDecimals(3)
        self.freq_input.setSuffix(" MHz")
        self.freq_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2C2C2C;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
            }
        """)
        control_panel.addWidget(self.freq_input)

        self.sample_rate_input = QtWidgets.QDoubleSpinBox()
        self.sample_rate_input.setRange(0.1, 3.2)
        self.sample_rate_input.setDecimals(1)
        self.sample_rate_input.setSuffix(" MHz")
        self.sample_rate_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2C2C2C;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
            }
        """)
        control_panel.addWidget(self.sample_rate_input)

        self.demod_combo = QtWidgets.QComboBox()
        self.demod_combo.addItems(["None", "AM", "FM", "USB", "LSB", "CW"])
        self.demod_combo.setStyleSheet("""
            QComboBox {
                background-color: #2C2C2C;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        control_panel.addWidget(self.demod_combo)

        layout.addLayout(control_panel)

        # Tabs for different analysis views
        self.analysis_tabs = QtWidgets.QTabWidget()
        self.analysis_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background-color: #2C2C2C;
                color: #FFFFFF;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3A3A3A;
            }
        """)
        layout.addWidget(self.analysis_tabs)

        # IQ Plot
        self.iq_plot = QChartView()
        self.iq_plot.setRenderHint(QtGui.QPainter.Antialiasing)
        self.iq_series = QScatterSeries()
        self.iq_series.setMarkerSize(3)
        self.iq_chart = QChart()
        self.iq_chart.addSeries(self.iq_series)
        self.iq_chart.createDefaultAxes()
        self.iq_chart.setTitle("IQ Plot")
        self.iq_chart.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#1E1E1E")))
        self.iq_chart.setTitleBrush(QtGui.QBrush(QtCore.Qt.white))
        self.iq_chart.legend().setLabelColor(QtCore.Qt.white)
        self.iq_plot.setChart(self.iq_chart)
        self.iq_plot.setMinimumSize(400, 300)
        self.analysis_tabs.addTab(self.iq_plot, "IQ Plot")

        # Spectrum Plot
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground('#1E1E1E')
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(color=(0, 255, 255), width=2))
        self.spectrum_plot.setLabel('left', "Power (dB)", color='#FFFFFF')
        self.spectrum_plot.setLabel('bottom', "Frequency (MHz)", color='#FFFFFF')
        self.analysis_tabs.addTab(self.spectrum_plot, "Spectrum")

        # Enhanced Waterfall Plot
        self.waterfall_plot = pg.PlotWidget()
        self.waterfall_plot.setBackground('#1E1E1E')
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        self.waterfall_plot.setLabel('left', "Time", color='#FFFFFF')
        self.waterfall_plot.setLabel('bottom', "Frequency (MHz)", color='#FFFFFF')
        
        # Enhanced colormap for waterfall
        colors = [
            (0, 0, 0),        # Black
            (0, 0, 128),      # Navy
            (0, 0, 255),      # Blue
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Yellow
            (255, 128, 0),    # Orange
            (255, 0, 0),      # Red
            (255, 255, 255)   # White
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.waterfall_image.setLookupTable(cmap.getLookupTable())
        
        self.analysis_tabs.addTab(self.waterfall_plot, "Waterfall")

        # Spectrogram Plot
        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setBackground('#1E1E1E')
        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.spectrogram_image)
        self.spectrogram_plot.setLabel('left', "Frequency (MHz)", color='#FFFFFF')
        self.spectrogram_plot.setLabel('bottom', "Time", color='#FFFFFF')
        
        # Enhanced colormap for spectrogram
        colors = [
            (0, 0, 0),        # Black
            (0, 0, 128),      # Navy
            (0, 0, 255),      # Blue
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Yellow
            (255, 128, 0),    # Orange
            (255, 0, 0),      # Red
            (255, 255, 255)   # White
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.spectrogram_image.setLookupTable(cmap.getLookupTable())
        self.analysis_tabs.addTab(self.spectrogram_plot, "Spectrogram")

        # Demodulated Signal Plot
        self.demod_plot = pg.PlotWidget()
        self.demod_plot.setBackground('#1E1E1E')
        self.demod_curve = self.demod_plot.plot(pen=pg.mkPen(color=(255, 255, 0), width=2))
        self.demod_plot.setLabel('left', "Amplitude", color='#FFFFFF')
        self.demod_plot.setLabel('bottom', "Time", color='#FFFFFF')
        self.analysis_tabs.addTab(self.demod_plot, "Demodulated Signal")

        # Constellation Diagram
        self.constellation_plot = pg.PlotWidget()
        self.constellation_plot.setBackground('#1E1E1E')
        self.constellation_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 255, 120))
        self.constellation_plot.addItem(self.constellation_scatter)
        self.constellation_plot.setLabel('left', "Q", color='#FFFFFF')
        self.constellation_plot.setLabel('bottom', "I", color='#FFFFFF')
        self.analysis_tabs.addTab(self.constellation_plot, "Constellation")

        # Signal Statistics
        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-family: Courier;
                border: none;
            }
        """)
        self.analysis_tabs.addTab(self.stats_text, "Signal Statistics")

        # Add FFT size control
        self.add_fft_size_control()

        # Add progress bar
        self.add_progress_bar()

    def add_fft_size_control(self):
        fft_layout = QtWidgets.QHBoxLayout()
        fft_label = QtWidgets.QLabel("FFT Size:")
        fft_label.setStyleSheet("color: #FFFFFF;")
        self.fft_size_combo = QtWidgets.QComboBox()
        self.fft_size_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.fft_size_combo.setCurrentText("1024")
        self.fft_size_combo.setStyleSheet("""
            QComboBox {
                background-color: #2C2C2C;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
            }
        """)
        self.fft_size_combo.currentTextChanged.connect(self.update_fft_size)
        fft_layout.addWidget(fft_label)
        fft_layout.addWidget(self.fft_size_combo)
        self.layout().insertLayout(1, fft_layout)

    def add_progress_bar(self):
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 1px;
            }
        """)
        self.layout().addWidget(self.progress_bar)

    def toggle_analysis(self):
        if self.is_processing:
            self.stop_analysis()
        else:
            self.start_analysis()

    def start_analysis(self):
        try:
            center_freq = self.freq_input.value() * 1e6
            sample_rate = self.sample_rate_input.value() * 1e6
            
            if not self.validate_input(center_freq, sample_rate):
                return

            self.sdr_controller.center_freq = center_freq
            self.sdr_controller.sample_rate = sample_rate
            
            if not self.sdr_controller.setup():
                raise Exception("Failed to set up SDR")

            self.is_processing = True
            self.start_button.setText("Stop Analysis")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                }
            """)
            
            self.processing_thread = threading.Thread(target=self.process_data)
            self.processing_thread.start()
        except Exception as e:
            self.error_signal.emit(f"Failed to start analysis: {str(e)}")

    def stop_analysis(self):
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
        self.start_button.setText("Start Analysis")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.sdr_controller.close()

    def process_data(self):
        while self.is_processing:
            try:
                samples = self.sdr_controller.read_samples()
                if samples is None:
                    continue

                self.last_samples = samples
                i_data = np.real(samples)
                q_data = np.imag(samples)
                
                freq_range, power_spectrum = self.sdr_controller.compute_power_spectrum(samples)
                if freq_range is None or power_spectrum is None:
                    continue

                # Update waterfall and spectrogram data
                self.waterfall_data.appendleft(power_spectrum)
                self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=0)
                self.spectrogram_data[-1, :] = power_spectrum

                demod_type = self.demod_combo.currentText()
                
                demodulated = self.demodulate(samples, demod_type)

                self.update_plots_signal.emit((i_data, q_data, freq_range, power_spectrum, demodulated))
                self.update_signal_statistics(samples, freq_range, power_spectrum)

                # Update progress bar (example: cycle between 0 and 100)
                self.update_progress((self.progress_bar.value() + 1) % 101)

            except Exception as e:
                self.error_signal.emit(f"Error in data processing: {str(e)}")
                traceback.print_exc()
                break

            # Add a small delay to prevent excessive CPU usage
            QtCore.QThread.msleep(10)

    def update_plots_gui(self, data):
        try:
            i_data, q_data, freq_range, power_spectrum, demodulated = data

            # Update IQ Plot
            self.iq_series.clear()
            iq_points = [QtCore.QPointF(i, q) for i, q in zip(i_data[:1000:10], q_data[:1000:10])]
            self.iq_series.append(iq_points)

            # Update Spectrum Plot
            freq_range_mhz = freq_range / 1e6
            self.spectrum_curve.setData(freq_range_mhz, power_spectrum)

            # Update Waterfall Plot
            waterfall_array = np.array(self.waterfall_data)
            self.waterfall_image.setImage(waterfall_array, autoLevels=False, levels=(-50, 0))

            # Update Spectrogram Plot
            self.spectrogram_image.setImage(self.spectrogram_data.T, autoLevels=False, levels=(-50, 0))

            # Update Demodulated Signal Plot
            self.demod_curve.setData(demodulated[:1000])

            # Update Constellation Diagram
            self.constellation_scatter.setData(i_data[:1000:10], q_data[:1000:10])

        except Exception as e:
            self.error_signal.emit(f"Error updating plots: {str(e)}")
            traceback.print_exc()

    def demodulate(self, samples, demod_type):
        if demod_type == "AM":
            return np.abs(samples)
        elif demod_type == "FM":
            return np.angle(samples[1:] * np.conj(samples[:-1]))
        elif demod_type == "USB":
            analytical_signal = signal.hilbert(np.real(samples))
            return np.real(analytical_signal)
        elif demod_type == "LSB":
            analytical_signal = signal.hilbert(np.real(samples))
            return np.real(analytical_signal * np.exp(-1j * 2 * np.pi * 0.25))
        elif demod_type == "CW":
            # Simple CW demodulation (envelope detection)
            return np.abs(signal.hilbert(np.real(samples)))
        else:
            return np.real(samples)

    def update_signal_statistics(self, samples, freq_range, power_spectrum):
        try:
            mean = np.mean(samples)
            std_dev = np.std(samples)
            max_amp = np.max(np.abs(samples))
            total_power = np.sum(np.abs(samples)**2)
            avg_power = total_power / len(samples)
            peak_freq = freq_range[np.argmax(power_spectrum)]
            bandwidth = self.estimate_bandwidth(freq_range, power_spectrum)
            noise_floor = np.median(power_spectrum)
            signal_power = np.max(power_spectrum)
            snr = 10 * np.log10(signal_power / noise_floor)
            phase = np.angle(samples)
            phase_mean = np.mean(phase)
            phase_std = np.std(phase)
            
            # Calculate modulation index (for FM signals)
            fm_demod = np.angle(samples[1:] * np.conj(samples[:-1]))
            mod_index = np.std(fm_demod)
            
            # Estimate symbol rate (if applicable)
            symbol_rate = self.estimate_symbol_rate(samples)
            
            stats_text = f"""
            Signal Statistics:
            ------------------
            Mean Amplitude:         {mean:.4f}
            Standard Deviation:     {std_dev:.4f}
            Max Amplitude:          {max_amp:.4f}
            Total Power:            {total_power:.4f}
            Average Power:          {avg_power:.4f}
            Peak Frequency:         {peak_freq/1e6:.4f} MHz
            Estimated Bandwidth:    {bandwidth/1e3:.2f} kHz
            Estimated SNR:          {snr:.2f} dB
            Mean Phase:             {phase_mean:.4f} rad
            Phase Standard Dev:     {phase_std:.4f} rad
            Modulation Index (FM):  {mod_index:.4f}
            Estimated Symbol Rate:  {symbol_rate:.2f} symbols/s
            """
            self.update_stats_signal.emit(stats_text)
        except Exception as e:
            self.error_signal.emit(f"Error updating statistics: {str(e)}")
            traceback.print_exc()

    def update_stats_gui(self, stats_text):
        self.stats_text.setText(stats_text)

    def estimate_bandwidth(self, freq_range, power_spectrum):
        peak_power = np.max(power_spectrum)
        threshold = peak_power - 3  # 3 dB bandwidth
        above_threshold = power_spectrum > threshold
        bandwidth = freq_range[above_threshold][-1] - freq_range[above_threshold][0]
        return bandwidth

    def estimate_symbol_rate(self, samples):
        # Estimate symbol rate using spectral analysis
        fft = np.fft.fft(np.abs(samples))
        freqs = np.fft.fftfreq(len(samples), 1/self.sdr_controller.sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        magnitude_spectrum = np.abs(fft[:len(fft)//2])
        
        # Find peaks in the magnitude spectrum
        peaks, _ = signal.find_peaks(magnitude_spectrum, height=np.max(magnitude_spectrum)/10)
        if len(peaks) > 1:
            # Estimate symbol rate as the difference between the two highest peaks
            sorted_peaks = sorted(peaks, key=lambda x: magnitude_spectrum[x], reverse=True)
            symbol_rate = np.abs(positive_freqs[sorted_peaks[0]] - positive_freqs[sorted_peaks[1]])
        else:
            symbol_rate = 0
        
        return symbol_rate

    def validate_input(self, center_freq, sample_rate):
        if center_freq < 24e6 or center_freq > 1766e6:
            self.error_signal.emit("Center frequency must be between 24 MHz and 1766 MHz.")
            return False
        if sample_rate < 0.1e6 or sample_rate > 3.2e6:
            self.error_signal.emit("Sample rate must be between 0.1 MHz and 3.2 MHz.")
            return False
        return True

    def update_fft_size(self, size):
        try:
            self.fft_size = int(size)
            self.sdr_controller.fft_size = self.fft_size
            self.spectrogram_data = np.zeros((256, self.fft_size))
        except Exception as e:
            self.error_signal.emit(f"Error updating FFT size: {str(e)}")
            traceback.print_exc()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_error_message(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        self.stop_analysis()
        super().closeEvent(event)

# End of SignalAnalysisTab class

# You may want to add any additional helper functions or classes here if needed