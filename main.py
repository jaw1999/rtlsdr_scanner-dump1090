# main.py
import sys
import warnings
import os
from PyQt5 import QtWidgets, QtGui, QtCore

# Enable high DPI scaling for better display on high-resolution screens
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# Optionally suppress RuntimeWarnings (comment out to see warnings)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import your custom tabs and SDR controller
from home_tab import HomeTab
from settings_tab import SettingsTab
from dump1090_tab import Dump1090Tab  # Ensure this file exists and is correctly implemented
from sdr_controller import SDRController  # Ensure this file exists and is correctly implemented

class SpectrumAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.sdr_controller = SDRController()  # Singleton instance
        self.initUI()
        self.connect_signals()

    def initUI(self):
        self.setWindowTitle("RTL-SDR Spectrum Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
                color: #ECEFF4;
            }
            QTabWidget::pane {
                border: 1px solid #4C566A;
                background-color: #3B4252;
            }
            QTabBar::tab {
                background-color: #4C566A;
                color: #ECEFF4;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #5E81AC;
            }
        """)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # Define JSON output directory for dump1090
        self.dump1090_json_dir = os.path.abspath("dump1090_json")
        os.makedirs(self.dump1090_json_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize Dump1090Tab first to prevent HomeTab from accessing SDR
        self.dump1090_tab = Dump1090Tab(self.sdr_controller, self.dump1090_json_dir)  # Pass JSON directory
        self.home_tab = HomeTab(self.sdr_controller)
        self.settings_tab = SettingsTab(self.sdr_controller)

        self.tabs.addTab(self.dump1090_tab, "Dump1090")
        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Add a status bar
        self.statusBar().showMessage("Ready")

    def connect_signals(self):
        # Connect signals from SDRController to methods that handle UI changes
        self.sdr_controller.dump1090_started.connect(self.on_dump1090_started)
        self.sdr_controller.dump1090_stopped.connect(self.on_dump1090_stopped)

    def on_dump1090_started(self):
        # Stop any ongoing SDR scan in HomeTab
        self.home_tab.stop_scan()
        # Disable SDR controls in HomeTab when Dump1090 is running
        self.home_tab.disable_sdr_controls()
        self.statusBar().showMessage("Dump1090 is running. SDR controls disabled.")

    def on_dump1090_stopped(self):
        # Enable SDR controls in HomeTab when Dump1090 has stopped
        self.home_tab.enable_sdr_controls()
        self.statusBar().showMessage("Dump1090 has stopped. SDR controls enabled.")

    def closeEvent(self, event):
        # Ensure SDR and Dump1090 are properly closed when the application is closed
        self.dump1090_tab.stop_dump1090()  # Ensure dump1090 is stopped
        self.sdr_controller.close()
        event.accept()

if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")  # Use Fusion style for a more modern look

        ex = SpectrumAnalyzer()
        ex.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application Error: {str(e)}")
