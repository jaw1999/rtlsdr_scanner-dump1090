# dump1090_tab.py
import os
import json
import logging
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Dump1090Tab(QtWidgets.QWidget):
    def __init__(self, sdr_controller, json_output_dir):
        super().__init__()
        self.sdr_controller = sdr_controller
        self.json_output_dir = json_output_dir
        self.dump1090_process = QtCore.QProcess(self)
        self.aircraft_json_path = os.path.join(self.json_output_dir, "aircraft.json")
        self.last_mtime = None  # To track last modification time
        self.initUI()
        self.setup_process_signals()

        # Configure logging
        logging.basicConfig(filename='dump1090.log', level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Control panel
        control_panel = QtWidgets.QHBoxLayout()
        self.start_stop_button = QtWidgets.QPushButton("Start Dump1090")
        self.start_stop_button.clicked.connect(self.toggle_dump1090)
        control_panel.addWidget(self.start_stop_button)
        layout.addLayout(control_panel)

        # Aircraft table
        self.aircraft_table = QtWidgets.QTableWidget()
        self.aircraft_table.setColumnCount(6)
        self.aircraft_table.setHorizontalHeaderLabels(['ICAO', 'Callsign', 'Altitude (ft)', 'Speed (kt)', 'Heading (°)', 'Last Seen (s)'])
        self.aircraft_table.horizontalHeader().setStretchLastSection(True)
        self.aircraft_table.setSortingEnabled(True)
        layout.addWidget(self.aircraft_table)

        # Search bar
        search_layout = QtWidgets.QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search Callsign or ICAO")
        self.search_input.textChanged.connect(self.filter_aircraft_table)
        search_layout.addWidget(QtWidgets.QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # Map view
        self.map_view = QWebEngineView()
        # Ensure 'map.html' is in the same directory as your script or provide the absolute path
        map_path = QtCore.QUrl.fromLocalFile(os.path.abspath("map.html"))
        self.map_view.setUrl(map_path)
        layout.addWidget(self.map_view)

        # Status label
        self.status_label = QtWidgets.QLabel("Dump1090 not running")
        layout.addWidget(self.status_label)

        # Timer for updating data
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.check_for_updates)
        self.update_timer.start(1000)  # Check every second

    def setup_process_signals(self):
        self.dump1090_process.readyReadStandardOutput.connect(self.handle_stdout)
        self.dump1090_process.readyReadStandardError.connect(self.handle_stderr)
        self.dump1090_process.started.connect(self.on_process_started)
        self.dump1090_process.finished.connect(self.on_process_finished)

    def toggle_dump1090(self):
        if self.dump1090_process.state() == QtCore.QProcess.NotRunning:
            # Ensure SDRController is not accessing the SDR device
            if self.sdr_controller.is_active:
                QtWidgets.QMessageBox.warning(self, "SDR Active", "Cannot start Dump1090 while SDR is active.")
                print("Cannot start Dump1090: SDR is active.")
                return
            self.start_dump1090()
        else:
            self.stop_dump1090()

    def start_dump1090(self):
        try:
            # Path to the local dump1090 executable
            dump1090_executable = os.path.join(os.path.abspath("dump1090"), "dump1090")
            if not os.path.isfile(dump1090_executable):
                raise Exception(f"dump1090 executable not found at {dump1090_executable}")

            # Ensure JSON output directory exists
            os.makedirs(self.json_output_dir, exist_ok=True)

            # Arguments: include --write-json
            arguments = [
                "--net",
                "--device-type", "rtlsdr",
                "--write-json", self.json_output_dir,
                "--write-json-every", "1"  # Write JSON every second
            ]

            # Start dump1090
            self.dump1090_process.start(dump1090_executable, arguments)

            if not self.dump1090_process.waitForStarted(5000):
                raise Exception("dump1090 failed to start.")

            self.start_stop_button.setText("Stop Dump1090")
            self.status_label.setText("Dump1090 running")
            self.sdr_controller.is_active = True  # Indicate that dump1090 is using the SDR
            self.sdr_controller.dump1090_started.emit()  # Emit signal
            logging.info("Dump1090 started.")
            print("Dump1090 started.")
        except Exception as e:
            self.status_label.setText(f"Error starting Dump1090: {str(e)}")
            logging.error(f"Error starting Dump1090: {str(e)}")
            print(f"Error starting Dump1090: {str(e)}")

    def stop_dump1090(self):
        if self.dump1090_process.state() != QtCore.QProcess.NotRunning:
            self.dump1090_process.terminate()
            if not self.dump1090_process.waitForFinished(5000):
                self.dump1090_process.kill()
            self.start_stop_button.setText("Start Dump1090")
            self.status_label.setText("Dump1090 stopped")
            self.sdr_controller.is_active = False  # Indicate that dump1090 has stopped using the SDR
            self.sdr_controller.dump1090_stopped.emit()  # Emit signal
            logging.info("Dump1090 stopped.")
            print("Dump1090 stopped.")

    def on_process_started(self):
        print("dump1090 started.")
        logging.info("dump1090 started.")

    def on_process_finished(self, exitCode, exitStatus):
        print(f"dump1090 finished with exit code {exitCode}, status {exitStatus}.")
        logging.info(f"dump1090 finished with exit code {exitCode}, status {exitStatus}.")
        self.start_stop_button.setText("Start Dump1090")
        self.status_label.setText("Dump1090 stopped")
        self.sdr_controller.is_active = False  # Ensure flag is reset
        self.sdr_controller.dump1090_stopped.emit()  # Emit signal
        logging.info("Dump1090 stopped.")
        print("Dump1090 stopped.")

    def handle_stdout(self):
        data = self.dump1090_process.readAllStandardOutput().data().decode()
        # Optionally process stdout data if needed
        print(data)
        logging.info(data)

    def handle_stderr(self):
        data = self.dump1090_process.readAllStandardError().data().decode()
        # Optionally process stderr data if needed
        print(f"dump1090 error: {data}")
        logging.error(f"dump1090 error: {data}")

    def check_for_updates(self):
        """
        Periodically check if aircraft.json has been updated and process it.
        """
        if not os.path.exists(self.aircraft_json_path):
            return  # File doesn't exist yet

        try:
            mtime = os.path.getmtime(self.aircraft_json_path)
            if self.last_mtime is None or mtime > self.last_mtime:
                self.last_mtime = mtime
                self.read_json_file()
        except Exception as e:
            self.status_label.setText(f"Error accessing JSON file: {str(e)}")
            logging.error(f"Error accessing JSON file: {str(e)}")
            print(f"Error accessing JSON file: {str(e)}")

    def read_json_file(self):
        """
        Read and parse the aircraft.json file, then update the UI.
        """
        try:
            print("Attempting to read aircraft.json")
            with open(self.aircraft_json_path, 'r') as f:
                data = json.load(f)
                aircraft = data.get('aircraft', [])
                print(f"Successfully read aircraft.json with {len(aircraft)} aircraft.")
                self.update_aircraft_table(aircraft)
                self.update_map(aircraft)
                self.status_label.setText(f"Fetched {len(aircraft)} aircraft.")
                logging.info("Fetched and updated aircraft data from JSON.")
                print("Fetched and updated aircraft data from JSON.")
        except json.JSONDecodeError as jde:
            self.status_label.setText(f"JSON Decode Error: {str(jde)}")
            logging.error(f"JSON Decode Error: {str(jde)}")
            print(f"JSON Decode Error: {str(jde)}")
        except Exception as e:
            self.status_label.setText(f"Error reading JSON: {str(e)}")
            logging.error(f"Error reading JSON: {str(e)}")
            print(f"Error reading JSON: {str(e)}")

    def update_aircraft_table(self, data):
        """
        Update the aircraft table with new data.
        """
        self.aircraft_table.setRowCount(len(data))
        print(f"Updating aircraft table with {len(data)} entries.")
        for row, aircraft in enumerate(data):
            hex_code = aircraft.get('hex', '')
            flight = aircraft.get('flight', '').strip()
            altitude = aircraft.get('alt_baro') or aircraft.get('alt_geom') or "N/A"
            speed = aircraft.get('gs', "N/A")
            heading = aircraft.get('track', "N/A")
            seen = aircraft.get('seen', "N/A")

            self.aircraft_table.setItem(row, 0, QtWidgets.QTableWidgetItem(hex_code))
            self.aircraft_table.setItem(row, 1, QtWidgets.QTableWidgetItem(flight))
            self.aircraft_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(altitude)))
            self.aircraft_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(speed)))
            self.aircraft_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(heading)))
            self.aircraft_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(seen)))
        print("Aircraft table updated successfully.")

    def update_map(self, data):
        """
        Send aircraft data to the map for rendering.
        """
        # Filter out aircraft without necessary data
        filtered_data = [
            {
                "hex": ac.get('hex', ''),
                "flight": ac.get('flight', '').strip(),
                "alt_baro": ac.get('alt_baro'),
                "alt_geom": ac.get('alt_geom'),
                "gs": ac.get('gs'),
                "track": ac.get('track'),
                "lat": ac.get('lat'),
                "lon": ac.get('lon')
            }
            for ac in data if ac.get('lat') and ac.get('lon')
        ]

        # Pass the data to the map via JavaScript
        js_command = f"updateAircraft({json.dumps(filtered_data)});"
        self.map_view.page().runJavaScript(js_command)
        print("Map updated with new aircraft positions.")
        logging.info("Map updated with new aircraft positions.")

    def filter_aircraft_table(self, text):
        """
        Filters the aircraft table based on the search input.
        Hides rows that do not match the search criteria.
        """
        for row in range(self.aircraft_table.rowCount()):
            match = False
            for column in [0, 1]:  # ICAO and Callsign columns
                item = self.aircraft_table.item(row, column)
                if item and text.lower() in item.text().lower():
                    match = True
                    break
            self.aircraft_table.setRowHidden(row, not match)

    def closeEvent(self, event):
        """
        Handle the widget close event to ensure dump1090 is stopped.
        """
        self.stop_dump1090()
        super().closeEvent(event)
