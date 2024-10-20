# Spectrum Analyzer with Dump1090 Integration

![Project Logo](icons/plane.png)

## Overview

**Spectrum Analyzer** is a desktop application built with PyQt5 that integrates with [Dump1090](https://github.com/antirez/dump1090) to visualize real-time aircraft data. The application features a user-friendly interface with a table displaying detailed aircraft information and an interactive map showing the positions and movements of multiple aircraft. Additionally, the **Home Tab** provides advanced SDR (Software Defined Radio) controls, spectrum analysis, and waterfall visualization for a comprehensive radio monitoring experience.

## Features

### Home Tab

- **Spectrum Plot:** Visualize the radio frequency spectrum in real-time, displaying signal strengths across various frequencies.
- **Waterfall Plot:** Monitor signal activity over time with a dynamic waterfall display, indicating the intensity and movement of signals.
- **SDR Controls:**
  - **Center Frequency (MHz):** Set the central frequency for monitoring.
  - **Sample Rate (MHz):** Adjust the sample rate to balance between frequency resolution and bandwidth.
  - **Gain:** Select from available gain settings to optimize signal reception.
  - **FFT Size:** Choose the FFT size for spectrum analysis (512, 1024, 2048, 4096).
  - **Averaging:** Configure the number of averages to smooth out the spectrum display.
  - **Color Scheme:** Select from various color schemes (Viridis, Plasma, Inferno, Magma, Cividis) for both spectrum and waterfall plots.
  - **Waterfall Speed:** Adjust the speed at which new data is added to the waterfall plot.
- **Controls:**
  - **Start/Stop Scan:** Begin or halt the SDR scanning process.
  - **Reset SDR:** Reset the SDR device to its default settings.
- **Status Indicators:** Real-time feedback on the SDR status and any operational messages.

### Dump1090 Tab

- **Control Dump1090:** Start and stop the Dump1090 process directly from the application.
- **Real-Time Aircraft Data:** Display live data including ICAO codes, callsigns, altitude, speed, heading, and last seen time.
- **Interactive Map:** Visualize aircraft positions on an interactive Leaflet map with rotated plane icons indicating direction.
- **Search and Filter:** Easily search for specific aircraft by callsign or ICAO code.
- **Automatic Updates:** The map and table update automatically as new data is received.
- **Error Handling and Logging:** Comprehensive logging for debugging and monitoring purposes.

## Demo

![Application Screenshot](screenshots/app_screenshot.png)

## Requirements

- Python 3.6 or higher
- [Dump1090](https://github.com/antirez/dump1090) installed and configured
- RTL-SDR dongle for receiving ADS-B signals

## Installation

1. **Clone the Repository:**

 
   git clone https://github.com/yourusername/spectrum-analyzer-dump1090.git
   cd spectrum-analyzer-dump1090

2. Set Up a Virtual Environment (Optional but Recommended):


    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Python Dependencies

    pip install -r requirements.txt

    
4. Ensure Dump1090 is Installed:

    Follow the Dump1090 installation guide to install and configure Dump1090 on your system.

5. Configure Dump1090 JSON Output:

    Ensure Dump1090 is configured to output JSON data to the dump1090_json directory. You can start Dump1090 with the following command:

    dump1090 --net --device-type rtlsdr --write-json dump1090_json --write-json-every 1


Usage

1. Run the Application:

    python main.py

2. Navigate Through Tabs:

    Home Tab:
        Configure SDR Settings: Adjust frequency, sample rate, gain, FFT size, averaging, color scheme, and waterfall speed.
        Start Scanning: Click the Start button to begin SDR scanning. The spectrum and waterfall plots will update in real-time.
        Stop Scanning: Click the Stop button to halt scanning.
        Reset SDR: Click the Reset SDR button to reset SDR settings to default.
    Dump1090 Tab:
        Start Dump1090: Click the Start Dump1090 button to launch Dump1090.
        Stop Dump1090: Click the Stop Dump1090 button to terminate the Dump1090 process.
        View Aircraft Data: Monitor the aircraft table and interactive map for real-time ADS-B data.

3. Interact with the Map:

    View Aircraft Positions: Multiple plane icons will appear on the map, each rotated based on their heading.
    View Details: Click on any plane icon to view detailed information about the aircraft.

4. Search and Filter:

    Use the search bar in the Dump1090 Tab to filter aircraft by callsign or ICAO code.


spectrum-analyzer-dump1090/
├── dump1090_tab.py          # PyQt5 tab handling Dump1090 integration
├── home_tab.py              # PyQt5 Home Tab with SDR controls and spectrum visualization
├── main.py                  # Main application entry point
├── map.html                 # HTML file for the interactive map
├── icons/
│   └── plane.png            # Custom plane icon for the map
├── plugins/
│   └── Leaflet.RotatedMarker.js  # Leaflet plugin for rotating markers
├── dump1090_json/           # Directory for Dump1090 JSON output (ignored in .gitignore)
├── dump1090.log             # Log file for Dump1090 and application logs
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
