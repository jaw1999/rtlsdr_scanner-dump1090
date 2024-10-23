# signal_classification_tab.py

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QTimer
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
import os
from classification_spectrum_widget import ClassificationSpectrumWidget
from classification_waterfall_widget import ClassificationWaterfallWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingWorker(QObject):
    """Worker class to handle model training in a separate thread."""
    training_progress = pyqtSignal(int)  # Emit training progress percentage
    training_finished = pyqtSignal(str)  # Emit when training is finished
    error_occurred = pyqtSignal(str)     # Emit when an error occurs
    training_metrics = pyqtSignal(float, float)  # Emit loss and accuracy

    def __init__(self, model, training_data, training_labels, epochs=10, batch_size=32):
        super().__init__()
        self.model = model
        self.training_data = training_data
        self.training_labels = training_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self._is_running = True

    def stop(self):
        """Stop the training process."""
        self._is_running = False

    def run(self):
        """Execute the training process."""
        try:
            # Define a callback to update progress and emit metrics
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, worker):
                    super().__init__()
                    self.worker = worker

                def on_epoch_end(self, epoch, logs=None):
                    if not self.worker._is_running:
                        self.model.stop_training = True
                        return
                    progress = int(((epoch + 1) / self.worker.epochs) * 100)
                    self.worker.training_progress.emit(progress)
                    self.worker.training_metrics.emit(
                        logs.get('loss', 0),
                        logs.get('accuracy', 0)
                    )

            # Start training with the callback
            self.model.fit(
                self.training_data,
                self.training_labels,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[ProgressCallback(self)],
                verbose=0
            )

            if self._is_running:
                self.training_finished.emit("Training completed successfully")
            else:
                self.training_finished.emit("Training stopped by user")

        except Exception as e:
            self.error_occurred.emit(str(e))


class SignalClassificationTab(QtWidgets.QWidget):
    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller

        # Initialize state variables
        self.model = None
        self.training_thread = None
        self.training_worker = None
        self.training_data = []
        self.training_labels = []
        self.is_training = False
        self.selected_region = None
        self.active_stream_tab = None  # To keep track of which tab is streaming

        # Setup logging
        logging.basicConfig(
            filename='signal_classification.log',
            level=logging.DEBUG,  # Set to DEBUG for detailed logs
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        logging.info("SignalClassificationTab initialized.")

        # Initialize UI
        self.init_ui()

        # Setup plot timer
        self.plot_timer = QTimer()
        self.plot_timer.setInterval(100)  # Update every 100ms
        self.plot_timer.timeout.connect(self.update_plots)

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create QTabWidget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #4C566A;
                background-color: #2E3440;
            }
            QTabBar::tab { 
                background: #3B4252;
                color: #ECEFF4;
                padding: 10px;
            }
            QTabBar::tab:selected { 
                background: #88C0D0;
                color: #2E3440;
            }
        """)
        main_layout.addWidget(self.tabs)

        # Create Tabs
        self.collect_samples_tab = self.create_collect_samples_tab()
        self.training_tab = self.create_training_tab()
        self.testing_tab = self.create_testing_tab()

        # Add Tabs to QTabWidget
        self.tabs.addTab(self.collect_samples_tab, "Collect Samples")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.testing_tab, "Testing")

        # Connect tab change to handle exclusive streaming
        self.tabs.currentChanged.connect(self.handle_tab_change)

        # Status bar
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #A3BE8C;
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
                background-color: #2E3440;
            }
        """)
        main_layout.addWidget(self.status_label)

    def handle_tab_change(self, index):
        """Handle exclusive streaming when switching tabs."""
        # Retrieve the name of the new tab
        new_tab_name = self.tabs.tabText(index)

        # If there's an active stream and it's not in the new tab, stop it
        if self.active_stream_tab and self.active_stream_tab != new_tab_name:
            logging.info(f"Switching from '{self.active_stream_tab}' to '{new_tab_name}'. Stopping active stream.")
            self.stop_streaming(self.active_stream_tab)
            self.active_stream_tab = None

    def create_collect_samples_tab(self):
        """Create the 'Collect Samples' tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # SDR Control Panel
        control_panel = self.create_control_panel(tab_name="Collect Samples")
        layout.addLayout(control_panel)

        # Splitter for Spectrum and Waterfall Displays
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setStyleSheet("background-color: #2E3440;")
        splitter.setHandleWidth(5)  # Thinner handle for aesthetics

        # Spectrum Display
        self.spectrum_widget_collect = ClassificationSpectrumWidget()
        self.spectrum_widget_collect.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.spectrum_widget_collect.region_selected.connect(self.on_region_selected)
        self.spectrum_widget_collect.setMinimumSize(400, 300)  # Adjust as needed
        splitter.addWidget(self.spectrum_widget_collect)

        # Waterfall Display
        self.waterfall_widget_collect = ClassificationWaterfallWidget()
        self.waterfall_widget_collect.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.waterfall_widget_collect.setMinimumSize(400, 300)  # Adjust as needed
        splitter.addWidget(self.waterfall_widget_collect)

        # Set initial sizes
        splitter.setSizes([1, 1])

        layout.addWidget(splitter)

        # Selection Buttons
        selection_layout = QtWidgets.QHBoxLayout()
        button_style = """
            QPushButton {
                min-width: 150px;
                height: 35px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 15px;
                margin: 2px;
                background-color: #A3BE8C;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #D8DEE9;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """
        self.label_signal_button_collect = QtWidgets.QPushButton("✓ Label as Signal")
        self.label_signal_button_collect.setStyleSheet(button_style)
        self.label_signal_button_collect.clicked.connect(lambda: self.label_selected_region(1))
        self.label_signal_button_collect.setEnabled(False)
        self.label_signal_button_collect.setToolTip("Label the selected region as Signal.")

        self.label_noise_button_collect = QtWidgets.QPushButton("✗ Label as Noise")
        self.label_noise_button_collect.setStyleSheet("""
            QPushButton {
                min-width: 150px;
                height: 35px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 15px;
                margin: 2px;
                background-color: #BF616A;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #D8DEE9;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """)
        self.label_noise_button_collect.clicked.connect(lambda: self.label_selected_region(0))
        self.label_noise_button_collect.setEnabled(False)
        self.label_noise_button_collect.setToolTip("Label the selected region as Noise.")

        selection_layout.addWidget(self.label_signal_button_collect)
        selection_layout.addWidget(self.label_noise_button_collect)
        selection_layout.addStretch()

        layout.addLayout(selection_layout)

        return tab

    def create_training_tab(self):
        """Create the 'Training' tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Training Parameters
        params_group = QtWidgets.QGroupBox("Training Parameters")
        params_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4C566A;
                border-radius: 8px;
                padding: 10px;
                background-color: #2E3440;
            }
            QGroupBox::title {
                color: #88C0D0;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        params_layout = QtWidgets.QGridLayout(params_group)

        param_style = """
            QLabel {
                color: #ECEFF4;
                font-size: 12px;
            }
            QSpinBox {
                background-color: #3B4252;
                color: #ECEFF4;
                border: 1px solid #4C566A;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }
        """

        params_layout.addWidget(QtWidgets.QLabel("Epochs:"), 0, 0)
        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(10)
        self.epochs_input.setStyleSheet(param_style)
        self.epochs_input.setToolTip("Number of training epochs.")
        params_layout.addWidget(self.epochs_input, 0, 1)

        params_layout.addWidget(QtWidgets.QLabel("Batch Size:"), 1, 0)
        self.batch_size_input = QtWidgets.QSpinBox()
        self.batch_size_input.setRange(1, 1024)
        self.batch_size_input.setValue(32)
        self.batch_size_input.setStyleSheet(param_style)
        self.batch_size_input.setToolTip("Size of training batches.")
        params_layout.addWidget(self.batch_size_input, 1, 1)

        layout.addWidget(params_group)

        # Training Control Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_style = """
            QPushButton {
                min-width: 120px;
                height: 35px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 15px;
                margin: 2px;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #D8DEE9;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """
        self.start_training_button = QtWidgets.QPushButton("▶ Start Training")
        self.start_training_button.setStyleSheet(button_style + """
            background-color: #A3BE8C;
            color: #FFFFFF;
        """)
        self.start_training_button.clicked.connect(self.start_training)
        self.start_training_button.setToolTip("Start training the model with collected samples.")
        button_layout.addWidget(self.start_training_button)

        self.stop_training_button = QtWidgets.QPushButton("⏹ Stop Training")
        self.stop_training_button.setStyleSheet(button_style + """
            background-color: #BF616A;
            color: #FFFFFF;
        """)
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setEnabled(False)
        self.stop_training_button.setToolTip("Stop the ongoing training process.")
        button_layout.addWidget(self.stop_training_button)

        self.save_model_button = QtWidgets.QPushButton("💾 Save Model")
        self.save_model_button.setStyleSheet(button_style + """
            background-color: #88C0D0;
            color: #FFFFFF;
        """)
        self.save_model_button.clicked.connect(self.save_model)
        self.save_model_button.setEnabled(False)
        self.save_model_button.setToolTip("Save the trained model to disk.")
        button_layout.addWidget(self.save_model_button)

        self.load_model_button = QtWidgets.QPushButton("📂 Load Model")
        self.load_model_button.setStyleSheet(button_style + """
            background-color: #81A1C1;
            color: #FFFFFF;
        """)
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setToolTip("Load a pre-trained model from disk.")
        button_layout.addWidget(self.load_model_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Training Metrics Plot
        metrics_group = QtWidgets.QGroupBox("Training Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4C566A;
                border-radius: 8px;
                padding: 10px;
                background-color: #2E3440;
            }
            QGroupBox::title {
                color: #88C0D0;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)

        self.training_figure = Figure(facecolor='#2E3440')
        self.training_canvas = FigureCanvas(self.training_figure)
        self.ax_training = self.training_figure.add_subplot(111)
        self.ax_training.set_facecolor('#2E3440')
        self.setup_training_plot()
        metrics_layout.addWidget(self.training_canvas)

        self.training_progress_bar = QtWidgets.QProgressBar()
        self.training_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4C566A;
                border-radius: 5px;
                text-align: center;
                color: #ECEFF4;
                background-color: #3B4252;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #A3BE8C;
                border-radius: 3px;
            }
        """)
        self.training_progress_bar.setValue(0)
        self.training_progress_bar.setToolTip("Training progress.")
        metrics_layout.addWidget(self.training_progress_bar)

        layout.addWidget(metrics_group)

        return tab

    def create_testing_tab(self):
        """Create the 'Testing' tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # SDR Control Panel for Testing
        control_panel = self.create_control_panel(tab_name="Testing")
        layout.addLayout(control_panel)

        # Splitter for Spectrum and Waterfall Displays
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setStyleSheet("background-color: #2E3440;")
        splitter.setHandleWidth(5)  # Thinner handle for aesthetics

        # Spectrum Display for Testing
        self.spectrum_widget_test = ClassificationSpectrumWidget()
        self.spectrum_widget_test.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.spectrum_widget_test.region_selected.connect(self.on_region_selected_test)
        self.spectrum_widget_test.setMinimumSize(400, 300)  # Adjust as needed
        splitter.addWidget(self.spectrum_widget_test)

        # Waterfall Display for Testing
        self.waterfall_widget_test = ClassificationWaterfallWidget()
        self.waterfall_widget_test.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.waterfall_widget_test.setMinimumSize(400, 300)  # Adjust as needed
        splitter.addWidget(self.waterfall_widget_test)

        # Set initial sizes
        splitter.setSizes([1, 1])

        layout.addWidget(splitter)

        # Prediction Controls
        prediction_layout = QtWidgets.QHBoxLayout()
        button_style = """
            QPushButton {
                min-width: 150px;
                height: 35px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 15px;
                margin: 2px;
                background-color: #88C0D0;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #D8DEE9;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """
        self.run_prediction_button = QtWidgets.QPushButton("🔍 Run Prediction")
        self.run_prediction_button.setStyleSheet(button_style)
        self.run_prediction_button.clicked.connect(self.run_prediction)
        self.run_prediction_button.setEnabled(False)
        self.run_prediction_button.setToolTip("Run the trained model against the selected region.")

        self.prediction_result_label = QtWidgets.QLabel("Prediction: N/A")
        self.prediction_result_label.setStyleSheet("""
            QLabel {
                color: #ECEFF4;
                font-size: 12px;
                background-color: #3B4252;
                padding: 5px;
                border-radius: 4px;
            }
        """)
        self.prediction_result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.prediction_result_label.setMinimumWidth(200)

        prediction_layout.addWidget(self.run_prediction_button)
        prediction_layout.addWidget(self.prediction_result_label)
        prediction_layout.addStretch()

        layout.addLayout(prediction_layout)

        # Load and Save Model Buttons in Testing Tab
        model_buttons_layout = QtWidgets.QHBoxLayout()
        model_button_style = """
            QPushButton {
                min-width: 120px;
                height: 35px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 15px;
                margin: 2px;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #D8DEE9;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """
        self.load_model_button_test = QtWidgets.QPushButton("📂 Load Model")
        self.load_model_button_test.setStyleSheet(model_button_style + """
            background-color: #81A1C1;
            color: #FFFFFF;
        """)
        self.load_model_button_test.clicked.connect(self.load_model)
        self.load_model_button_test.setToolTip("Load a pre-trained model from disk.")
        model_buttons_layout.addWidget(self.load_model_button_test)

        self.save_model_button_test = QtWidgets.QPushButton("💾 Save Model")
        self.save_model_button_test.setStyleSheet(model_button_style + """
            background-color: #88C0D0;
            color: #FFFFFF;
        """)
        self.save_model_button_test.clicked.connect(self.save_model)
        self.save_model_button_test.setEnabled(False)
        self.save_model_button_test.setToolTip("Save the trained model to disk.")
        model_buttons_layout.addWidget(self.save_model_button_test)

        model_buttons_layout.addStretch()

        layout.addLayout(model_buttons_layout)

        return tab

    def create_control_panel(self, tab_name):
        """Create the SDR control panel for a given tab."""
        control_layout = QtWidgets.QHBoxLayout()

        # Style for control dropdowns
        combo_style = """
            QComboBox {
                background-color: #3B4252;
                color: #ECEFF4;
                border: 1px solid #4C566A;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3B4252;
                color: #ECEFF4;
                selection-background-color: #88C0D0;
                selection-color: #2E3440;
            }
        """

        # Style for control buttons
        button_style = """
            QPushButton {
                min-width: 120px;
                height: 35px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 15px;
                margin: 2px;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #D8DEE9;
            }
        """

        # SDR Controls
        self.start_stream_button = QtWidgets.QPushButton("▶ Start")
        self.start_stream_button.setStyleSheet(button_style + """
            background-color: #A3BE8C;
            color: #FFFFFF;
        """)
        self.start_stream_button.clicked.connect(lambda: self.start_streaming(tab_name))
        self.start_stream_button.setToolTip("Start streaming data from the SDR device.")

        self.stop_stream_button = QtWidgets.QPushButton("⏹ Stop")
        self.stop_stream_button.setStyleSheet(button_style + """
            background-color: #BF616A;
            color: #FFFFFF;
        """)
        self.stop_stream_button.clicked.connect(lambda: self.stop_streaming(tab_name))
        self.stop_stream_button.setEnabled(False)
        self.stop_stream_button.setToolTip("Stop streaming data from the SDR device.")

        self.reset_button = QtWidgets.QPushButton("↺ Reset SDR")
        self.reset_button.setStyleSheet(button_style + """
            background-color: #88C0D0;
            color: #FFFFFF;
        """)
        self.reset_button.clicked.connect(lambda: self.reset_sdr(tab_name))
        self.reset_button.setToolTip("Reset the SDR device to default settings.")

        # SDR Parameters
        param_layout = QtWidgets.QGridLayout()
        param_layout.setSpacing(5)

        # Frequency control
        param_layout.addWidget(QtWidgets.QLabel("Frequency (MHz):"), 0, 0)
        self.freq_input = QtWidgets.QLineEdit(str(self.sdr_controller.center_freq / 1e6))
        self.freq_input.setStyleSheet(combo_style)
        self.freq_input.setToolTip("Set the center frequency in MHz.")
        param_layout.addWidget(self.freq_input, 0, 1)

        # Sample rate control
        param_layout.addWidget(QtWidgets.QLabel("Sample Rate (MHz):"), 1, 0)
        self.sample_rate_input = QtWidgets.QLineEdit(str(self.sdr_controller.sample_rate / 1e6))
        self.sample_rate_input.setStyleSheet(combo_style)
        self.sample_rate_input.setToolTip("Set the sample rate in MHz.")
        param_layout.addWidget(self.sample_rate_input, 1, 1)

        # Gain control (Removed "auto" option to prevent NoneType)
        param_layout.addWidget(QtWidgets.QLabel("Gain:"), 2, 0)
        self.gain_combo = QtWidgets.QComboBox()
        self.gain_combo.setStyleSheet(combo_style)
        # Removed the line: self.gain_combo.addItem("auto")
        self.gain_combo.addItems([str(g) for g in self.sdr_controller.get_available_gains()])
        self.gain_combo.setCurrentIndex(0)  # Set to first gain value by default
        self.gain_combo.setToolTip("Select the gain value.")
        param_layout.addWidget(self.gain_combo, 2, 1)

        # Add controls to main control layout
        control_layout.addWidget(self.start_stream_button)
        control_layout.addWidget(self.stop_stream_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addLayout(param_layout)
        control_layout.addStretch()

        return control_layout

    def setup_training_plot(self):
        """Initialize the training metrics plot."""
        self.ax_training.clear()
        self.ax_training.set_title("Training Metrics", color='#ECEFF4')
        self.ax_training.set_xlabel("Epoch", color='#ECEFF4')
        self.ax_training.set_ylabel("Value", color='#ECEFF4')
        self.ax_training.grid(True, color='#4C566A', alpha=0.5)

        # Initialize empty lines for loss and accuracy
        self.loss_line, = self.ax_training.plot([], [], label='Loss', color='#BF616A')
        self.acc_line, = self.ax_training.plot([], [], label='Accuracy', color='#A3BE8C')
        self.ax_training.legend(facecolor='#3B4252', labelcolor='#ECEFF4')

        self.training_metrics = {'loss': [], 'accuracy': []}
        self.training_canvas.draw()
        logging.info("Training plot initialized.")

    def start_streaming(self, tab_name):
        """Start SDR streaming for a given tab."""
        try:
            # Ensure exclusive streaming
            if self.active_stream_tab and self.active_stream_tab != tab_name:
                self.stop_streaming(self.active_stream_tab)

            if not self.sdr_controller.is_active:
                # Get parameters from UI
                center_freq_text = self.freq_input.text()
                sample_rate_text = self.sample_rate_input.text()
                gain_text = self.gain_combo.currentText()

                # Validate and convert frequency and sample rate
                try:
                    center_freq = float(center_freq_text) * 1e6
                    sample_rate = float(sample_rate_text) * 1e6
                except ValueError:
                    raise ValueError("Frequency and Sample Rate must be valid numbers.")

                # Validate frequency and sample rate ranges
                if not (1e6 <= center_freq <= 1.7e9):
                    raise ValueError("Center frequency must be between 1 MHz and 1.7 GHz")
                if not (1e6 <= sample_rate <= 3.2e6):
                    raise ValueError("Sample rate must be between 1 MHz and 3.2 MHz")

                # Configure SDR
                self.sdr_controller.center_freq = center_freq
                self.sdr_controller.sample_rate = sample_rate
                # Ensure gain is always a float (remove "auto" option)
                if gain_text.lower() == 'auto':
                    # Since SDRController does not handle 'auto', set to a default gain, e.g., 20.0 dB
                    self.sdr_controller.gain = 20.0
                    logging.warning("Auto gain selected, defaulting to 20.0 dB.")
                else:
                    self.sdr_controller.gain = float(gain_text)

                # Setup SDR
                success = self.sdr_controller.setup()
                if not success:
                    raise RuntimeError("Failed to initialize SDR device")

            # Start streaming
            self.plot_timer.start()
            self.active_stream_tab = tab_name
            self.start_stream_button.setEnabled(False)
            self.stop_stream_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.status_label.setText(f"Streaming active in '{tab_name}' tab.")
            self.status_label.setStyleSheet("color: #A3BE8C;")
            logging.info(f"Started streaming in '{tab_name}' tab.")

        except ValueError as ve:
            QtWidgets.QMessageBox.warning(
                self,
                "Input Validation Error",
                f"{str(ve)}"
            )
            logging.error(f"Input Validation Error: {str(ve)}")
        except Exception as e:
            error_msg = f"Streaming Error: {str(e)}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: #BF616A;")
            logging.error(error_msg)
            QtWidgets.QMessageBox.critical(
                self,
                "Streaming Error",
                f"Failed to start streaming:\n{str(e)}"
            )

    def stop_streaming(self, tab_name):
        """Stop SDR streaming for a given tab."""
        if self.active_stream_tab == tab_name:
            self.plot_timer.stop()
            self.active_stream_tab = None
            self.start_stream_button.setEnabled(True)
            self.stop_stream_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.sdr_controller.close()
            self.status_label.setText(f"Streaming stopped in '{tab_name}' tab.")
            self.status_label.setStyleSheet("color: #EBCB8B;")
            logging.info(f"Stopped streaming in '{tab_name}' tab.")

    def update_plots(self):
        """Update spectrum and waterfall plots with latest data."""
        try:
            # Get samples from SDR
            samples = self.sdr_controller.read_samples()
            logging.debug("read_samples called.")
            logging.debug(f"Raw Samples: {samples}")

            if samples is None or len(samples) == 0:
                raise ValueError("No samples received from SDR.")

            # Compute power spectrum
            freq_range, power_spectrum_dbm = self.sdr_controller.compute_power_spectrum(samples)
            logging.debug("compute_power_spectrum called.")
            logging.debug(f"Frequency Range: {freq_range}")
            logging.debug(f"Power Spectrum dBm: {power_spectrum_dbm}")

            if freq_range is None or power_spectrum_dbm is None:
                raise ValueError("Failed to compute power spectrum.")

            # Update Spectrum and Waterfall Widgets based on the active tab
            if self.active_stream_tab == "Collect Samples":
                self.spectrum_widget_collect.update_plot(freq_range, power_spectrum_dbm)
                self.waterfall_widget_collect.update_plot(freq_range, power_spectrum_dbm)
            elif self.active_stream_tab == "Testing":
                self.spectrum_widget_test.update_plot(freq_range, power_spectrum_dbm)
                self.waterfall_widget_test.update_plot(freq_range, power_spectrum_dbm)

        except Exception as e:
            error_msg = f"Plot Error: {str(e)}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: #BF616A;")
            self.stop_streaming(self.active_stream_tab)
            logging.error(f"Error updating plots: {str(e)}")

    def on_region_selected(self, start_freq, end_freq):
        """Handle region selection from spectrum widget in Collect Samples tab."""
        self.selected_region = (start_freq, end_freq)
        self.label_signal_button_collect.setEnabled(True)
        self.label_noise_button_collect.setEnabled(True)
        self.status_label.setText(f"Selected: {start_freq:.2f} MHz to {end_freq:.2f} MHz")
        self.status_label.setStyleSheet("color: #EBCB8B;")
        logging.info(f"Region selected from {start_freq} MHz to {end_freq} MHz.")

    def on_region_selected_test(self, start_freq, end_freq):
        """Handle region selection from spectrum widget in Testing tab."""
        self.selected_region = (start_freq, end_freq)
        self.run_prediction_button.setEnabled(True)
        self.status_label.setText(f"Selected: {start_freq:.2f} MHz to {end_freq:.2f} MHz")
        self.status_label.setStyleSheet("color: #EBCB8B;")
        logging.info(f"Region selected in Testing tab from {start_freq} MHz to {end_freq} MHz.")

    def label_selected_region(self, label):
        """Label the selected region as signal or noise."""
        if self.selected_region is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Selection Required",
                "Please select a region in the spectrum plot first."
            )
            return

        try:
            # Get selected data
            freqs, powers = self.spectrum_widget_collect.get_selected_data()
            if freqs is None or powers is None:
                raise ValueError("No data available in selected region")

            # Extract features
            features = self.extract_features(powers)

            # Add to training data
            self.training_data.append(features)
            self.training_labels.append(label)

            # Update UI
            self.status_label.setText(
                f"Labeled as {'Signal' if label == 1 else 'Noise'} "
                f"(Total samples: {len(self.training_data)})"
            )
            self.status_label.setStyleSheet("color: #A3BE8C;")
            logging.info(f"Labeled region as {'Signal' if label == 1 else 'Noise'}. Total samples: {len(self.training_data)}.")

            # Reset selection state
            self.spectrum_widget_collect.clear_selection()
            self.label_signal_button_collect.setEnabled(False)
            self.label_noise_button_collect.setEnabled(False)
            self.selected_region = None

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Labeling Error",
                f"Error labeling region: {str(e)}"
            )
            logging.error(f"Error labeling region: {str(e)}")

    def extract_features(self, powers):
        """Extract features from the power spectrum data."""
        # Normalize the power values
        normalized = powers - np.mean(powers)
        if np.max(np.abs(normalized)) > 0:
            normalized = normalized / np.max(np.abs(normalized))

        # Calculate additional features
        features = []

        # Basic statistical features
        features.extend([
            np.mean(powers),           # Mean power
            np.std(powers),            # Standard deviation
            np.max(powers),            # Peak power
            np.min(powers),            # Minimum power
            np.median(powers),         # Median power
            np.percentile(powers, 75), # 75th percentile
            np.percentile(powers, 25), # 25th percentile
        ])

        # Spectral features
        fft_vals = np.fft.fft(normalized)
        fft_freqs = np.fft.fftfreq(len(normalized))
        fft_magnitude = np.abs(fft_vals)

        features.extend([
            np.sum(fft_magnitude),     # Total spectral power
            np.max(fft_magnitude),     # Peak spectral component
            np.mean(fft_magnitude),    # Average spectral power
            np.std(fft_magnitude),     # Spectral spread
        ])

        # Shape features
        gradient = np.gradient(normalized)
        features.extend([
            np.mean(np.abs(gradient)), # Average slope
            np.max(np.abs(gradient)),  # Maximum slope
            np.std(gradient),          # Slope variation
        ])

        logging.debug(f"Extracted features: {features}")
        return np.array(features)

    def start_training(self):
        """Start the model training process."""
        if len(self.training_data) < 10:
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient Data",
                "Please collect at least 10 samples before training."
            )
            return

        try:
            # Prepare training data
            X = np.array(self.training_data)
            y = np.array(self.training_labels)

            # Shuffle the data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Create or ensure model exists
            if self.model is None:
                self.model = self.create_model(input_shape=X.shape[1])

            # Reset training metrics
            self.training_metrics = {'loss': [], 'accuracy': []}
            self.setup_training_plot()

            # Create and configure training worker
            self.training_worker = TrainingWorker(
                model=self.model,
                training_data=X,
                training_labels=y,
                epochs=self.epochs_input.value(),
                batch_size=self.batch_size_input.value()
            )

            # Setup training thread
            self.training_thread = QThread()
            self.training_worker.moveToThread(self.training_thread)

            # Connect signals
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.training_progress.connect(self.update_training_progress)
            self.training_worker.training_finished.connect(self.on_training_finished)
            self.training_worker.error_occurred.connect(self.on_training_error)
            self.training_worker.training_metrics.connect(self.update_training_metrics)

            # Update UI
            self.is_training = True
            self.start_training_button.setEnabled(False)
            self.stop_training_button.setEnabled(True)
            self.save_model_button.setEnabled(False)
            self.save_model_button_test.setEnabled(False)
            self.status_label.setText("Training in progress...")
            self.status_label.setStyleSheet("color: #EBCB8B;")
            logging.info("Started training.")

            # Start training
            self.training_thread.start()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Training Error",
                f"Error starting training: {str(e)}"
            )
            logging.error(f"Error starting training: {str(e)}")
            self.reset_training_state()

    def create_model(self, input_shape):
        """Create the neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        logging.info(f"Model created with input shape: {input_shape}")
        return model

    def stop_training(self):
        """Stop the model training process."""
        if self.training_worker and self.is_training:
            self.training_worker.stop()
            self.training_thread.quit()
            self.training_thread.wait()
            self.reset_training_state()
            self.status_label.setText("Training stopped by user")
            self.status_label.setStyleSheet("color: #EBCB8B;")
            logging.info("Training stopped by user.")

    def update_training_progress(self, progress):
        """Update the training progress bar."""
        self.training_progress_bar.setValue(progress)
        logging.debug(f"Training progress updated to {progress}%.")
        if progress >= 100:
            self.training_progress_bar.setValue(100)

    def update_training_metrics(self, loss, accuracy):
        """Update the training metrics plot."""
        self.training_metrics['loss'].append(loss)
        self.training_metrics['accuracy'].append(accuracy)

        epochs = range(1, len(self.training_metrics['loss']) + 1)

        self.loss_line.set_data(epochs, self.training_metrics['loss'])
        self.acc_line.set_data(epochs, self.training_metrics['accuracy'])

        self.ax_training.set_xlim(1, max(epochs))
        self.ax_training.set_ylim(0, max(
            max(self.training_metrics['loss']), 
            max(self.training_metrics['accuracy'])
        ) * 1.1)

        self.training_canvas.draw()
        logging.debug(f"Training metrics updated: Epoch {epochs[-1]}, Loss {loss}, Accuracy {accuracy}")

    def reset_training_state(self):
        """Reset the training-related UI elements."""
        self.is_training = False
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        self.save_model_button.setEnabled(True if self.model else False)
        self.save_model_button_test.setEnabled(True if self.model else False)
        logging.info("Training state reset.")

    def on_training_finished(self, message):
        """Handle training completion."""
        self.reset_training_state()
        self.save_model_button.setEnabled(True)
        self.save_model_button_test.setEnabled(True)
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #A3BE8C;")
        logging.info(message)

        # Show completion message with final metrics
        if self.training_metrics['accuracy']:
            final_accuracy = self.training_metrics['accuracy'][-1]
            QtWidgets.QMessageBox.information(
                self,
                "Training Complete",
                f"Training completed successfully.\nFinal accuracy: {final_accuracy:.2f}"
            )

    def on_training_error(self, error_message):
        """Handle training errors."""
        self.reset_training_state()
        self.status_label.setText(f"Training error: {error_message}")
        self.status_label.setStyleSheet("color: #BF616A;")

        QtWidgets.QMessageBox.critical(
            self,
            "Training Error",
            f"An error occurred during training:\n{error_message}"
        )
        logging.error(f"Training error: {error_message}")

    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Model",
                "No trained model available to save."
            )
            return

        try:
            # Get save path from user
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Model",
                "",
                "H5 Files (*.h5);;All Files (*)"
            )

            if file_path:
                # Ensure .h5 extension
                if not file_path.endswith('.h5'):
                    file_path += '.h5'

                # Save model
                self.model.save(file_path)

                # Also save feature extraction parameters
                metadata_path = file_path.replace('.h5', '_metadata.json')
                metadata = {
                    'feature_count': self.model.get_layer(index=0).input_shape[1],
                    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'sample_count': len(self.training_data)
                }

                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f)

                self.status_label.setText(f"Model saved to {file_path}")
                self.status_label.setStyleSheet("color: #A3BE8C;")

                QtWidgets.QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Model and metadata saved successfully to:\n{file_path}"
                )
                logging.info(f"Model saved to {file_path}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"Error saving model: {str(e)}"
            )
            logging.error(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load a trained model from disk."""
        try:
            # Get load path from user
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load Model",
                "",
                "H5 Files (*.h5);;All Files (*)"
            )

            if file_path:
                # Load model
                self.model = tf.keras.models.load_model(file_path)

                # Try to load metadata if it exists
                metadata_path = file_path.replace('.h5', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        import json
                        metadata = json.load(f)
                        feature_count = metadata.get('feature_count')
                        if feature_count != self.model.get_layer(index=0).input_shape[1]:
                            raise ValueError("Model feature count doesn't match current feature extraction")

                self.status_label.setText(f"Model loaded from {file_path}")
                self.status_label.setStyleSheet("color: #A3BE8C;")
                self.save_model_button.setEnabled(True)
                self.save_model_button_test.setEnabled(True)

                QtWidgets.QMessageBox.information(
                    self,
                    "Load Successful",
                    f"Model loaded successfully from:\n{file_path}"
                )
                logging.info(f"Model loaded from {file_path}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Error loading model: {str(e)}"
            )
            logging.error(f"Error loading model: {str(e)}")

    def run_prediction(self):
        """Run model prediction on the current spectrum data in Testing tab."""
        try:
            # Get current data from Spectrum widget in Testing tab
            freqs, powers = self.spectrum_widget_test.get_selected_data()
            if freqs is None or powers is None:
                raise ValueError("No data available for prediction. Please select a region in the spectrum plot.")

            # Predict
            prediction = self.predict_signal(powers)
            if prediction is not None:
                prediction_label = "Signal" if prediction >= 0.5 else "Noise"
                self.prediction_result_label.setText(f"Prediction: {prediction_label} ({prediction:.2f})")
                logging.info(f"Prediction made: {prediction_label} ({prediction:.2f})")
            else:
                self.prediction_result_label.setText("Prediction: N/A")
                logging.warning("Prediction returned None.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Prediction Error",
                f"Error during prediction: {str(e)}"
            )
            logging.error(f"Prediction error: {str(e)}")

    def predict_signal(self, powers):
        """Predict whether a signal is present in the given power spectrum."""
        if self.model is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Model",
                "No trained model available. Please train a model first."
            )
            return None

        try:
            # Extract features from the power spectrum
            features = self.extract_features(powers)

            # Reshape for prediction
            features = features.reshape(1, -1)

            # Make prediction
            prediction = self.model.predict(features)[0][0]

            logging.debug(f"Prediction made: {prediction}")
            return prediction

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return None

    def reset_sdr(self, tab_name):
        """Reset the SDR device for a given tab."""
        try:
            if self.active_stream_tab == tab_name:
                self.stop_streaming(tab_name)

            success = self.sdr_controller.reset()
            if success:
                self.status_label.setText("SDR Reset Successfully")
                self.status_label.setStyleSheet("color: #A3BE8C;")
                logging.info(f"SDR reset successfully in '{tab_name}' tab.")
            else:
                raise RuntimeError("Failed to reset SDR device")

        except Exception as e:
            error_msg = f"SDR Reset Error: {str(e)}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: #BF616A;")
            logging.error(error_msg)
            QtWidgets.QMessageBox.critical(
                self,
                "SDR Reset Error",
                f"Failed to reset SDR device:\n{str(e)}\n\nPlease disconnect and reconnect the device."
            )

    def closeEvent(self, event):
        """Handle cleanup when closing the tab."""
        # Stop streaming if active
        if self.plot_timer.isActive():
            self.stop_streaming(self.active_stream_tab)

        # Stop training if active
        if self.is_training:
            self.stop_training()

        # Close SDR connection
        self.sdr_controller.close()

        # Log closure
        logging.info("Signal classification tab closed.")

        event.accept()
