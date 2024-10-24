from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import logging
import os
import json
import datetime
import pywt  # For wavelet features
import scipy.stats  # For statistical features
from models.signal_model import SignalModel
from widgets.classification_spectrum_widget import ClassificationSpectrumWidget
from widgets.classification_waterfall_widget import ClassificationWaterfallWidget

class StyleSheet:
    """Central place for all stylesheet definitions"""
    
    DARK_THEME = """
    QWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Segoe UI', Arial;
    }
    
    QTabWidget::pane {
        border: 1px solid #3d3d3d;
        background-color: #1e1e1e;
    }
    
    QTabBar::tab {
        background-color: #2d2d2d;
        color: #e0e0e0;
        padding: 8px 20px;
        border: 1px solid #3d3d3d;
        border-bottom: none;
    }
    
    QTabBar::tab:selected {
        background-color: #3d4450;
    }
    
    QTableWidget {
        background-color: #2d2d2d;
        alternate-background-color: #353535;
        gridline-color: #4a4a4a;
        border: 1px solid #4a4a4a;
        selection-background-color: #3d4450;
    }
    
    QTableWidget::item {
        padding: 5px;
    }
    
    QHeaderView::section {
        background-color: #383838;
        padding: 8px;
        border: 1px solid #4a4a4a;
        font-weight: bold;
    }
    
    QLineEdit {
        background-color: #2d2d2d;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        padding: 5px;
        selection-background-color: #3d4450;
    }
    
    QLineEdit:focus {
        border: 1px solid #5e81ac;
    }
    
    QPushButton {
        background-color: #5e81ac;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    QPushButton:hover {
        background-color: #81a1c1;
    }
    
    QPushButton:pressed {
        background-color: #4c566a;
    }
    
    QPushButton:disabled {
        background-color: #4a4a4a;
        color: #808080;
    }
    
    QProgressBar {
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        text-align: center;
    }
    
    QProgressBar::chunk {
        background-color: #5e81ac;
    }
    
    QLabel {
        color: #e0e0e0;
    }
    
    QStatusBar {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    
    QSpinBox {
        background-color: #2d2d2d;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        padding: 5px;
    }
    
    QComboBox {
        background-color: #2d2d2d;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        padding: 5px;
    }
    
    QComboBox::drop-down {
        border: none;
    }
    
    QComboBox::down-arrow {
        image: url(down_arrow.png);
        width: 12px;
        height: 12px;
    }
    
    QScrollBar:vertical {
        background-color: #2d2d2d;
        width: 12px;
        margin: 0px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #4a4a4a;
        min-height: 20px;
        border-radius: 6px;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QDialog {
        background-color: #1e1e1e;
    }
    """

class RecordingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording Settings")
        self.setStyleSheet(StyleSheet.DARK_THEME)
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Create form layout for inputs
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)
        
        # Signal Name
        self.signal_name = QtWidgets.QLineEdit()
        self.signal_name.setPlaceholderText("Enter signal name...")
        form_layout.addRow("Signal Name:", self.signal_name)
        
        # Duration
        self.duration = QtWidgets.QSpinBox()
        self.duration.setRange(1, 300)
        self.duration.setValue(10)
        self.duration.setSuffix(" seconds")
        form_layout.addRow("Duration:", self.duration)
        
        # Number of recordings
        self.num_recordings = QtWidgets.QSpinBox()
        self.num_recordings.setRange(1, 50)
        self.num_recordings.setValue(1)
        self.num_recordings.setSuffix(" recordings")
        form_layout.addRow("Batch Size:", self.num_recordings)
        
        layout.addLayout(form_layout)
        
        # Add buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.record_btn = QtWidgets.QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.record_btn)
        
        layout.addLayout(button_layout)

class ClassificationDialog(QtWidgets.QDialog):
    def __init__(self, prediction, matching_signals, confidence, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classification Results")
        self.setStyleSheet(StyleSheet.DARK_THEME)
        self.setup_ui(prediction, matching_signals, confidence)

    def setup_ui(self, prediction, matching_signals, confidence):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Prediction result
        result_layout = QtWidgets.QHBoxLayout()
        result_label = QtWidgets.QLabel("Prediction:")
        result_value = QtWidgets.QLabel(prediction)
        result_value.setStyleSheet("font-weight: bold; color: #88c0d0;")
        result_layout.addWidget(result_label)
        result_layout.addWidget(result_value)
        layout.addLayout(result_layout)
        
        # Confidence
        conf_layout = QtWidgets.QHBoxLayout()
        conf_label = QtWidgets.QLabel("Confidence:")
        conf_value = QtWidgets.QLabel(f"{confidence:.1f}%")
        conf_value.setStyleSheet("font-weight: bold; color: #88c0d0;")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(conf_value)
        layout.addLayout(conf_layout)
        
        # Matching signals
        if matching_signals:
            layout.addWidget(QtWidgets.QLabel("Matching known signals:"))
            for signal in matching_signals:
                signal_label = QtWidgets.QLabel(f"• {signal}")
                signal_label.setStyleSheet("color: #a3be8c;")
                layout.addWidget(signal_label)
        else:
            no_match = QtWidgets.QLabel("No matching known signals found")
            no_match.setStyleSheet("color: #bf616a;")
            layout.addWidget(no_match)
        
        # Close button
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

class RecordingsTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # Set columns
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels([
            "ID", "Name", "Frequency (MHz)", 
            "Duration (s)", "Timestamp", 
            "SNR (dB)", "Actions"
        ])
        
        # Set table properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        # Set header properties
        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

    def add_recording(self, recording_data):
        row = self.rowCount()
        self.insertRow(row)
        
        # Add recording details
        self.setItem(row, 0, QtWidgets.QTableWidgetItem(str(recording_data['id'])))
        self.setItem(row, 1, QtWidgets.QTableWidgetItem(recording_data['name']))
        self.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{recording_data['frequency']:.2f}"))
        self.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{recording_data['duration']:.1f}"))
        self.setItem(row, 4, QtWidgets.QTableWidgetItem(recording_data['timestamp']))
        self.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{recording_data['snr']:.1f}"))
        
        # Add action buttons
        actions_widget = QtWidgets.QWidget()
        actions_layout = QtWidgets.QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(4, 4, 4, 4)
        actions_layout.setSpacing(8)
        
        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.setFixedWidth(70)
        delete_btn.clicked.connect(lambda: self.parent().delete_recording(recording_data['id']))
        
        play_btn = QtWidgets.QPushButton("Play")
        play_btn.setFixedWidth(70)
        play_btn.clicked.connect(lambda: self.parent().play_recording(recording_data['id']))
        
        actions_layout.addWidget(play_btn)
        actions_layout.addWidget(delete_btn)
        
        self.setCellWidget(row, 6, actions_widget)

class SignalClassificationTab(QtWidgets.QWidget):
    def __init__(self, sdr_controller):
        super().__init__()
        self.sdr_controller = sdr_controller
        
        # Initialize state
        self.signal_model = SignalModel()
        self.is_recording = False
        self.current_recording = None
        self.recordings = {}
        self.recording_batch = []
        self.batch_count = 0
        self.total_batches = 0
        
        # Setup directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.recordings_dir = os.path.join(self.base_dir, "recordings")
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize UI
        self.init_ui()
        self.apply_dark_theme()
        
        # Load existing recordings
        self.load_recordings()
        
        # Setup timers
        self.recording_timer = QtCore.QTimer()
        self.recording_timer.timeout.connect(self.update_recording)
        
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(100)
        self.plot_timer.timeout.connect(self.update_plots)

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Analysis tab
        self.analysis_tab = QtWidgets.QWidget()
        analysis_layout = QtWidgets.QVBoxLayout(self.analysis_tab)
        
        # Add spectrum and waterfall displays
        self.spectrum_widget = ClassificationSpectrumWidget()
        self.waterfall_widget = ClassificationWaterfallWidget()
        
        analysis_layout.addWidget(self.spectrum_widget)
        analysis_layout.addWidget(self.waterfall_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        analysis_layout.addLayout(control_panel)
        
        # Action buttons
        action_layout = QtWidgets.QHBoxLayout()
        
        self.record_btn = QtWidgets.QPushButton("Record Signal")
        self.record_btn.clicked.connect(self.show_recording_dialog)
        self.record_btn.setEnabled(False)
        
        self.classify_btn = QtWidgets.QPushButton("Classify Signal")
        self.classify_btn.clicked.connect(self.classify_selection)
        self.classify_btn.setEnabled(False)
        
        action_layout.addWidget(self.record_btn)
        action_layout.addWidget(self.classify_btn)
        analysis_layout.addLayout(action_layout)
        
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        
        # Recordings tab
        self.recordings_tab = QtWidgets.QWidget()
        recordings_layout = QtWidgets.QVBoxLayout(self.recordings_tab)
        
        self.recordings_table = RecordingsTable(self)
        recordings_layout.addWidget(self.recordings_table)
        
        # Recording controls
        recording_controls = QtWidgets.QHBoxLayout()
        
        self.train_btn = QtWidgets.QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        
        self.save_model_btn = QtWidgets.QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        
        self.load_model_btn = QtWidgets.QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        
        recording_controls.addWidget(self.train_btn)
        recording_controls.addWidget(self.save_model_btn)
        recording_controls.addWidget(self.load_model_btn)
        recordings_layout.addLayout(recording_controls)
        
        self.tab_widget.addTab(self.recordings_tab, "Recordings")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_bar = QtWidgets.QStatusBar()
        layout.addWidget(self.status_bar)

    def apply_dark_theme(self):
        """Apply dark theme to all widgets."""
        self.setStyleSheet(StyleSheet.DARK_THEME)
        
        # Set dark theme for matplotlib plots
        for widget in [self.spectrum_widget, self.waterfall_widget]:
            widget.figure.patch.set_facecolor('#1e1e1e')
            widget.ax.set_facecolor('#2d2d2d')
            widget.ax.tick_params(colors='#e0e0e0')
            for spine in widget.ax.spines.values():
                spine.set_color('#4a4a4a')
            widget.canvas.draw()

    def create_control_panel(self):
        """Create the SDR control panel."""
        control_layout = QtWidgets.QGridLayout()
        control_layout.setSpacing(10)
        
        # Frequency controls
        freq_label = QtWidgets.QLabel("Center Frequency (MHz):")
        self.freq_input = QtWidgets.QLineEdit(str(self.sdr_controller.center_freq / 1e6))
        control_layout.addWidget(freq_label, 0, 0)
        control_layout.addWidget(self.freq_input, 0, 1)
        
        # Sample rate controls
        rate_label = QtWidgets.QLabel("Sample Rate (MHz):")
        self.sample_rate_input = QtWidgets.QLineEdit(str(self.sdr_controller.sample_rate / 1e6))
        control_layout.addWidget(rate_label, 1, 0)
        control_layout.addWidget(self.sample_rate_input, 1, 1)
        
        # Gain controls
        gain_label = QtWidgets.QLabel("Gain (dB):")
        self.gain_input = QtWidgets.QLineEdit(str(self.sdr_controller.gain))
        control_layout.addWidget(gain_label, 2, 0)
        control_layout.addWidget(self.gain_input, 2, 1)
        
        # Stream controls
        stream_layout = QtWidgets.QHBoxLayout()
        
        self.start_stream_btn = QtWidgets.QPushButton("Start Stream")
        self.start_stream_btn.clicked.connect(self.start_streaming)
        
        self.stop_stream_btn = QtWidgets.QPushButton("Stop Stream")
        self.stop_stream_btn.clicked.connect(self.stop_streaming)
        self.stop_stream_btn.setEnabled(False)
        
        stream_layout.addWidget(self.start_stream_btn)
        stream_layout.addWidget(self.stop_stream_btn)
        
        control_layout.addLayout(stream_layout, 3, 0, 1, 2)
        
        return control_layout

    def show_recording_dialog(self):
        """Show the recording settings dialog."""
        if not self.spectrum_widget.has_selection():
            self.status_bar.showMessage("Please select a frequency region first")
            return
            
        dialog = RecordingDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.start_recording_batch(
                dialog.signal_name.text(),
                dialog.duration.value(),
                dialog.num_recordings.value()
            )

    def start_recording_batch(self, signal_name, duration, num_recordings):
        """Initialize and start a batch recording session."""
        if not signal_name:
            self.status_bar.showMessage("Please enter a signal name")
            return
            
        self.recording_batch = []
        self.batch_count = 0
        self.total_batches = num_recordings
        
        # Store batch settings
        self.batch_settings = {
            'name': signal_name,
            'duration': duration,
            'num_recordings': num_recordings
        }
        
        # Start first recording
        self.start_recording()

    def start_recording(self):
        """Start recording the current signal."""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.batch_count += 1
        
        # Get frequency bounds
        start_freq, end_freq = self.spectrum_widget.get_selection_bounds()
        
        # Initialize new recording
        self.current_recording = {
            'id': len(self.recordings) + 1,
            'name': self.batch_settings['name'],
            'frequency': (start_freq + end_freq) / 2,
            'bandwidth': end_freq - start_freq,
            'duration': self.batch_settings['duration'],
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'samples': [],
            'snr': 0
        }
        
        # Show progress dialog
        self.progress_dialog = QtWidgets.QProgressDialog(
            f"Recording {self.batch_count} of {self.total_batches}...",
            "Cancel", 0, self.batch_settings['duration'] * 10, self)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.canceled.connect(self.stop_recording)
        
        # Start recording timer
        self.recording_timer.start(100)  # 100ms intervals
        self.status_bar.showMessage(f"Recording {self.batch_count} of {self.total_batches}")

    def update_recording(self):
        """Update the current recording progress."""
        if not self.is_recording:
            return
            
        # Get samples from SDR
        samples = self.sdr_controller.read_samples()
        if samples is not None:
            self.current_recording['samples'].extend(samples)
            
        # Update progress
        progress = len(self.current_recording['samples']) / (self.sdr_controller.sample_rate * self.current_recording['duration'])
        self.progress_dialog.setValue(int(progress * 100))
        
        # Check if recording is complete
        if progress >= 1:
            self.stop_recording()

    def stop_recording(self):
        """Stop the current recording."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.recording_timer.stop()
        
        # Calculate SNR
        samples = np.array(self.current_recording['samples'])
        signal_power = np.mean(np.abs(samples)**2)
        noise_floor = np.min(np.abs(samples)**2)
        self.current_recording['snr'] = 10 * np.log10(signal_power / noise_floor)
        
        # Save recording
        self.save_recording(self.current_recording)
        self.recording_batch.append(self.current_recording)
        
        self.progress_dialog.close()
        
        # Check if batch is complete
        if self.batch_count < self.total_batches:
            # Start next recording
            self.start_recording()
        else:
            self.status_bar.showMessage("Recording batch complete")

    def save_recording(self, recording):
        """Save the recording data and metadata."""
        try:
            # Save samples
            samples_path = os.path.join(
                self.recordings_dir, 
                f"recording_{recording['id']}.npy"
            )
            np.save(samples_path, np.array(recording['samples']))
            
            # Update metadata
            recording_copy = recording.copy()
            recording_copy['samples'] = samples_path
            self.recordings[recording['id']] = recording_copy
            
            # Update UI
            self.recordings_table.add_recording(recording_copy)
            self.save_recordings_metadata()
            
        except Exception as e:
            self.status_bar.showMessage(f"Error saving recording: {str(e)}")
            logging.error(f"Error saving recording: {str(e)}")

    def delete_recording(self, recording_id):
        """Delete a recording."""
        try:
            # Delete samples file
            samples_path = self.recordings[recording_id]['samples']
            if os.path.exists(samples_path):
                os.remove(samples_path)
            
            # Remove from recordings dict
            del self.recordings[recording_id]
            
            # Update UI
            for row in range(self.recordings_table.rowCount()):
                if int(self.recordings_table.item(row, 0).text()) == recording_id:
                    self.recordings_table.removeRow(row)
                    break
            
            self.save_recordings_metadata()
            self.status_bar.showMessage("Recording deleted")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error deleting recording: {str(e)}")
            logging.error(f"Error deleting recording: {str(e)}")

    def play_recording(self, recording_id):
        """Play back a recorded signal."""
        try:
            # Load samples
            samples_path = self.recordings[recording_id]['samples']
            samples = np.load(samples_path)
            
            # TODO: Implement signal playback functionality
            self.status_bar.showMessage("Playback not implemented yet")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error playing recording: {str(e)}")
            logging.error(f"Error playing recording: {str(e)}")

    def train_model(self):
        """Train the model using recorded signals."""
        if len(self.recordings) < 2:
            self.status_bar.showMessage("Need at least 2 recordings to train model")
            return
            
        try:
            # Prepare training data
            X = []
            y = []
            unique_signals = sorted(list(set(r['name'] for r in self.recordings.values())))
            
            progress = QtWidgets.QProgressDialog(
                "Preparing training data...",
                "Cancel", 0, len(self.recordings), self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            
            for i, recording in enumerate(self.recordings.values()):
                # Update progress
                progress.setValue(i)
                if progress.wasCanceled():
                    return
                
                # Load samples
                samples = np.load(recording['samples'])
                
                # Extract features
                features = self.extract_features(samples)
                X.append(features)
                
                # Convert signal name to numeric label
                label = unique_signals.index(recording['name'])
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            progress.setLabelText("Training model...")
            progress.setValue(0)
            progress.setMaximum(0)
            
            # Create and train model
            input_shape = X.shape[1]
            self.signal_model.create_model(input_shape)
            self.signal_model.train_model(X, y)
            
            progress.close()
            
            self.save_model_btn.setEnabled(True)
            self.status_bar.showMessage("Model trained successfully")
            
        except Exception as e:
            self.status_bar.showMessage(f"Training error: {str(e)}")
            logging.error(f"Training error: {str(e)}")

    def save_model(self):
        """Save the trained model."""
        try:
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Model",
                self.models_dir,
                "Model Files (*.h5)")
                
            if file_name:
                self.signal_model.save_model(file_name)
                self.status_bar.showMessage(f"Model saved to {file_name}")
                
        except Exception as e:
            self.status_bar.showMessage(f"Error saving model: {str(e)}")
            logging.error(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load a saved model."""
        try:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load Model",
                self.models_dir,
                "Model Files (*.h5)")
                
            if file_name:
                self.signal_model.load_model(file_name)
                self.save_model_btn.setEnabled(True)
                self.classify_btn.setEnabled(True)
                self.status_bar.showMessage(f"Model loaded from {file_name}")
                
        except Exception as e:
            self.status_bar.showMessage(f"Error loading model: {str(e)}")
            logging.error(f"Error loading model: {str(e)}")

    def start_streaming(self):
        """Start SDR streaming."""
        try:
            # Update SDR parameters
            self.sdr_controller.center_freq = float(self.freq_input.text()) * 1e6
            self.sdr_controller.sample_rate = float(self.sample_rate_input.text()) * 1e6
            self.sdr_controller.gain = float(self.gain_input.text())
            
            if self.sdr_controller.setup():
                self.plot_timer.start()
                self.start_stream_btn.setEnabled(False)
                self.stop_stream_btn.setEnabled(True)
                self.record_btn.setEnabled(True)
                self.classify_btn.setEnabled(True)
                self.status_bar.showMessage("Streaming started")
            else:
                self.status_bar.showMessage("Failed to start SDR")
                
        except Exception as e:
            self.status_bar.showMessage(f"Error starting stream: {str(e)}")
            logging.error(f"Error starting stream: {str(e)}")

    def stop_streaming(self):
        """Stop SDR streaming."""
        self.plot_timer.stop()
        self.sdr_controller.close()
        self.start_stream_btn.setEnabled(True)
        self.stop_stream_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.classify_btn.setEnabled(False)
        self.status_bar.showMessage("Streaming stopped")

    def update_plots(self):
        """Update spectrum and waterfall plots."""
        try:
            samples = self.sdr_controller.read_samples()
            if samples is None:
                return
                
            freq_range, power_spectrum = self.sdr_controller.compute_power_spectrum(samples)
            if freq_range is None or power_spectrum is None:
                return
                
            self.spectrum_widget.update_plot(freq_range, power_spectrum)
            self.waterfall_widget.update_plot(freq_range, power_spectrum)
            
        except Exception as e:
            self.status_bar.showMessage(f"Plot update error: {str(e)}")
            logging.error(f"Plot update error: {str(e)}")

    def save_recordings_metadata(self):
        """Save recordings metadata to JSON file."""
        try:
            metadata_path = os.path.join(self.recordings_dir, "recordings_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.recordings, f, indent=2)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving metadata: {str(e)}")
            logging.error(f"Error saving recordings metadata: {str(e)}")

    def load_recordings(self):
        """Load existing recordings metadata."""
        try:
            metadata_path = os.path.join(self.recordings_dir, "recordings_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.recordings = json.load(f)
                    
                # Verify recordings files exist and update table
                valid_recordings = {}
                for recording_id, recording in self.recordings.items():
                    if os.path.exists(recording['samples']):
                        valid_recordings[recording_id] = recording
                        self.recordings_table.add_recording(recording)
                    else:
                        logging.warning(f"Recording file not found: {recording['samples']}")
                
                self.recordings = valid_recordings
                self.save_recordings_metadata()  # Save cleaned metadata
                
        except Exception as e:
            self.status_bar.showMessage(f"Error loading recordings: {str(e)}")
            logging.error(f"Error loading recordings: {str(e)}")

    def extract_features(self, samples):
        """Extract features from signal samples for classification."""
        # Convert to numpy array if needed
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
            
        # Normalize samples
        samples = samples - np.mean(samples)
        samples = samples / (np.std(samples) + 1e-10)
        
        # Time domain features
        time_features = [
            np.mean(np.abs(samples)),  # Mean amplitude
            np.std(samples),           # Standard deviation
            np.max(np.abs(samples)),   # Peak amplitude
            np.sum(samples**2),        # Energy
            np.sum(np.abs(samples)),   # L1 norm
            scipy.stats.kurtosis(np.real(samples)),  # Kurtosis
            scipy.stats.skew(np.real(samples))       # Skewness
        ]
        
        # Frequency domain features
        fft = np.fft.fft(samples)
        fft_mag = np.abs(fft)
        fft_phase = np.angle(fft)
        
        freq_features = [
            np.mean(fft_mag),         # Mean magnitude
            np.std(fft_mag),          # Spectral spread
            np.max(fft_mag),          # Peak magnitude
            np.sum(fft_mag**2),       # Spectral energy
            np.mean(fft_phase),       # Mean phase
            np.std(fft_phase)         # Phase spread
        ]
        
        # Spectral entropy
        psd = fft_mag**2
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Wavelet features
        coeffs = pywt.wavedec(samples, 'db4', level=4)
        wavelet_features = [np.std(c) for c in coeffs]
        
        # Combine all features
        features = np.concatenate([
            time_features,
            freq_features,
            [spectral_entropy],
            wavelet_features
        ])
        
        return features

    def closeEvent(self, event):
        """Handle cleanup when closing the tab."""
        try:
            # Stop streaming if active
            if self.plot_timer.isActive():
                self.stop_streaming()
            
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()
            
            # Save recordings metadata
            self.save_recordings_metadata()
            
            event.accept()
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            event.accept()

    def classify_selection(self):
        """Classify the currently selected region in the spectrum."""
        if not self.spectrum_widget.has_selection():
            self.status_bar.showMessage("Please select a region to classify")
            return
            
        if not hasattr(self.signal_model, 'model') or self.signal_model.model is None:
            self.status_bar.showMessage("No model loaded. Please train or load a model first.")
            return
            
        try:
            # Get selected region data
            start_freq, end_freq = self.spectrum_widget.get_selection_bounds()
            samples = self.sdr_controller.read_samples()
            
            if samples is None:
                raise ValueError("No samples received from SDR")
                
            # Extract features
            features = self.extract_features(samples)
            
            # Get prediction
            prediction_label = self.signal_model.predict_signal(features)
            
            # Find matching signals
            matching_signals = []
            center_freq = (start_freq + end_freq) / 2
            bandwidth = end_freq - start_freq
            
            for recording in self.recordings.values():
                if abs(recording['frequency'] - center_freq) < bandwidth/2:
                    matching_signals.append(recording['name'])
                    
            matching_signals = list(set(matching_signals))
            
            # Show classification dialog
            dialog = ClassificationDialog(
                prediction=prediction_label,
                matching_signals=matching_signals,
                confidence=90.5,  # TODO: Get actual confidence from model
                parent=self
            )
            dialog.exec_()
            
        except Exception as e:
            self.status_bar.showMessage(f"Classification error: {str(e)}")
            logging.error(f"Classification error: {str(e)}")

if __name__ == '__main__':
    import sys
    from src.signal_processing.sdr_controller import SDRController
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application-wide style
    app.setStyle('Fusion')
    app.setStyleSheet(StyleSheet.DARK_THEME)
    
    # Create main window
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Signal Classification")
    
    # Create SDR controller
    sdr_controller = SDRController()
    
    # Create and set central widget
    central_widget = SignalClassificationTab(sdr_controller)
    main_window.setCentralWidget(central_widget)
    
    # Show window
    main_window.show()
    
    sys.exit(app.exec_())