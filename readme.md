# Spectrum Analyzer with Dump1090, Frequency Scanner, and LoRa Detection Integration

![Project Logo](icons/plane.png)

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
   - [Home Tab](#home-tab)
   - [Frequency Scanner Tab](#frequency-scanner-tab)
   - [Dump1090 Tab](#dump1090-tab)
   - [Signal Analysis Tab](#signal-analysis-tab)
   - [LoRa Detection Tab](#lora-detection-tab)
3. [Demo](#demo)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)


## Overview

**Spectrum Analyzer** is a sophisticated desktop application built with PyQt5 that seamlessly integrates with [Dump1090](https://github.com/antirez/dump1090) and includes a powerful Frequency Scanner and LoRa Detection capabilities. The application features a user-friendly interface with multiple tabs, each offering unique functionality:

1. A **Home Tab** with advanced SDR (Software Defined Radio) controls, spectrum analysis, and waterfall visualization.
2. A **Frequency Scanner Tab** that automatically scans a range of frequencies to detect active signals and logs them.
3. A **Dump1090 Tab** displaying detailed aircraft information in a table format and an interactive map showing the positions and movements of multiple aircraft.
4. A **Signal Analysis Tab** providing comprehensive signal processing capabilities, including IQ plots, spectrograms, and demodulation options.
5. A **LoRa Detection Tab** for detecting and analyzing LoRa signals across multiple frequency bands with detailed signal information.

This tool is designed for radio enthusiasts, IoT developers, aviation buffs, and anyone interested in real-time signal analysis, aircraft tracking, and LoRa signal detection.

## Features

### Home Tab

- **Spectrum Plot:** Visualize the radio frequency spectrum in real-time, displaying signal strengths across various frequencies.
- **Waterfall Plot:** Monitor signal activity over time with a dynamic waterfall display, indicating the intensity and movement of signals.
- **SDR Controls:**
  - **Center Frequency (MHz):** Set the central frequency for monitoring.
  - **Sample Rate (MHz):** Adjust the sample rate to balance between frequency resolution and bandwidth.
  - **Gain:** Select from available gain settings to optimize signal reception, including an auto gain option.
  - **FFT Size:** Choose the FFT size for spectrum analysis (512, 1024, 2048, 4096).
  - **Averaging:** Configure the number of averages to smooth out the spectrum display.
  - **Color Scheme:** Select from various color schemes (Viridis, Plasma, Inferno, Magma, Cividis) for both spectrum and waterfall plots.
  - **Waterfall Speed:** Adjust the speed at which new data is added to the waterfall plot.
- **Controls:**
  - **Start/Stop Scan:** Begin or halt the SDR scanning process.
  - **Reset SDR:** Reset the SDR device to its default settings.
- **Status Indicators:** Real-time feedback on the SDR status and any operational messages.

### Frequency Scanner Tab

- **Automatic Frequency Scanning:** Scan a defined frequency range to detect active signals.
- **Power Measurement in dBm:** Display signal power levels in dBm for accurate readings.
- **Threshold Settings:** Adjust the detection threshold to filter out noise.
- **Signal Logging:** Log detected frequencies and power levels for analysis.
- **User Interface Enhancements:**
  - **Start/Stop Scan:** Control the scanning process.
  - **Frequency Range Inputs:** Set the start and end frequencies for scanning.
  - **Scan Step Size:** Define the frequency increment for each scan step.
  - **Threshold Adjustment:** Set the power threshold for signal detection.
  - **Calibration Offset:** Adjust the calibration offset to fine-tune power measurements.
- **Visualization:**
  - **Real-Time Spectrum Display:** View the spectrum as the scanner sweeps through frequencies.
  - **Detected Signals List:** See a live list of detected signals with frequency and power information.

### Dump1090 Tab

- **Control Dump1090:** Start and stop the Dump1090 process directly from the application.
- **Real-Time Aircraft Data:** Display live data including ICAO codes, callsigns, altitude, speed, heading, and last seen time.
- **Interactive Map:** Visualize aircraft positions on an interactive Leaflet map with rotated plane icons indicating direction.
- **Search and Filter:** Easily search for specific aircraft by callsign or ICAO code.
- **Automatic Updates:** The map and table update automatically as new data is received.
- **Error Handling and Logging:** Comprehensive logging for debugging and monitoring purposes.

### Signal Analysis Tab

- **IQ Plot:** Visualize the In-phase and Quadrature components of the received signal.
- **Spectrum Analysis:** Advanced spectrum visualization with adjustable parameters.
- **Waterfall Display:** Enhanced waterfall plot with customizable color schemes.
- **Spectrogram:** 2D time-frequency representation of the signal.
- **Demodulation Options:** Support for various demodulation types (AM, FM, USB, LSB, CW).
- **Constellation Diagram:** Visualize digital modulation schemes.
- **Signal Statistics:** Real-time calculation and display of key signal parameters.
- **FFT Size Control:** Adjustable FFT size for detailed analysis.
- **Calibration Offset Control:** Adjust the calibration offset for accurate power measurements in dBm.

### LoRa Detection Tab

- **Real-Time LoRa Signal Detection:**
  - **Automatic Signal Detection:** Detect LoRa signals using correlation with reference chirp signals
  - **Multi-band Support:** Monitor common LoRa frequency bands (433 MHz, 868 MHz, 915 MHz)
  - **Configurable Parameters:** Adjust spreading factor, bandwidth, and detection threshold
  - **Signal Quality Metrics:** Monitor SNR, signal strength, and detection confidence
  
- **Visualization:**
  - **Spectrum Display:** Real-time spectrum visualization with peak hold capability
  - **Waterfall Display:** Time-frequency visualization of LoRa signals
  - **Color Scheme Selection:** Multiple color schemes for optimal visualization
  
- **Detection Information:**
  - **Frequency Display:** Show exact frequency of detected signals
  - **Signal Strength:** Display power levels in dBm
  - **Bandwidth:** Show configured bandwidth
  - **Spreading Factor:** Display current spreading factor
  - **SNR:** Show signal-to-noise ratio
  - **Detection History:** Track detection count and timing
  
- **Controls:**
  - **Preset Frequencies:** Quick selection of common LoRa frequencies
  - **Gain Control:** Adjustable gain settings including auto gain
  - **Peak Hold:** Enable/disable peak hold for spectrum display
  - **Detection Parameters:** Configurable detection threshold and signal parameters

## Demo

![Application Screenshot](screenshots/app_screenshot.png)

## Requirements

- Python 3.6 or higher
- [Dump1090](https://github.com/antirez/dump1090) installed and configured
- RTL-SDR dongle for receiving ADS-B signals and spectrum analysis

**Additional Dependencies:**

- PyQt5
- numpy
- pyrtlsdr
- pyqtgraph
- scipy
- matplotlib
- pandas
- requests

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/jaw1999/rtlsdr_scanner-dump1090.git
    cd spectrum-analyzer-dump1090
    ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure Dump1090 is Installed:**

    Follow the [Dump1090 installation guide](https://github.com/antirez/dump1090)

5. **Configure Dump1090 JSON Output:**

    Ensure Dump1090 is configured to output JSON data to the `dump1090_json` directory. You can start Dump1090 with the following command:

    ```bash
    dump1090 --net --device-type rtlsdr --write-json dump1090_json --write-json-every 1
    ```

## Usage

1. **Run the Application:**

    ```bash
    python main.py
    ```

2. **Navigate Through Tabs:**

    - **Home Tab:**
      - **Configure SDR Settings:** Adjust frequency, sample rate, gain (including auto gain), FFT size, averaging, color scheme, and waterfall speed.
      - **Start Scanning:** Click the "Start" button to begin SDR scanning. The spectrum and waterfall plots will update in real-time.
      - **Stop Scanning:** Click the "Stop" button to halt scanning.
      - **Reset SDR:** Click the "Reset SDR" button to reset SDR settings to default.

    - **Frequency Scanner Tab:**
      - **Set Scanning Parameters:** Define the start and end frequencies, scan step size, threshold, and calibration offset.
      - **Start Scanning:** Click the "Start Scan" button to begin scanning the frequency range.
      - **View Detected Signals:** Monitor the list of detected frequencies and their power levels.
      - **Stop Scanning:** Click the "Stop Scan" button to halt the scanning process.

    - **Dump1090 Tab:**
      - **Start Dump1090:** Click the "Start Dump1090" button to launch Dump1090.
      - **Stop Dump1090:** Click the "Stop Dump1090" button to terminate the Dump1090 process.
      - **View Aircraft Data:** Monitor the aircraft table and interactive map for real-time ADS-B data.

    - **Signal Analysis Tab:**
      - **Select Demodulation Type:** Choose from AM, FM, USB, LSB, or CW.
      - **Adjust FFT Size:** Select the desired FFT size for analysis.
      - **Adjust Calibration Offset:** Fine-tune the calibration offset for accurate power measurements.
      - **Monitor Plots:** Observe the IQ Plot, Spectrum, Waterfall, Spectrogram, and Constellation Diagram.
      - **View Signal Statistics:** Check real-time signal parameters and statistics.

    - **LoRa Detection Tab:**
      - **Select Frequency Band:** Choose from preset LoRa frequencies or enter a custom frequency.
      - **Configure LoRa Parameters:** Set spreading factor (7-12), bandwidth (7.8-500 kHz), and detection threshold.
      - **Start Detection:** Click "Start Detection" to begin monitoring for LoRa signals.
      - **Monitor Signals:** Watch the spectrum and waterfall displays for LoRa activity.
      - **View Detection Info:** Check the detection panel for detailed signal information.
      - **Enable Peak Hold:** Toggle peak hold to track maximum signal levels.
      - **Adjust Settings:** Fine-tune gain, sample rate, and FFT size for optimal detection.

3. **Interact with the Map:**

    - **View Aircraft Positions:** Multiple plane icons will appear on the map, each rotated based on their heading.
    - **View Details:** Click on any plane icon to view detailed information about the aircraft.

4. **Search and Filter:**

    - Use the search bar in the **Dump1090 Tab** to filter aircraft by callsign or ICAO code.
