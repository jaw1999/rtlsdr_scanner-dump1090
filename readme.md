# RTL-SDR Signal Analysis & Classification Tool

## Overview

A sophisticated desktop application built with PyQt5 that provides real-time signal analysis, visualization, and machine learning-based classification capabilities using RTL-SDR devices. The application features multiple specialized tabs for different types of signal analysis and processing.

## Features

### Analysis Tab
- Real-time spectrum visualization with interactive frequency selection
- Advanced waterfall display showing signal history over time
- Dark theme optimized interface with adjustable scales
- Mouse-based region selection for signal analysis
- Scrollable power level adjustments
- Real-time power measurements in dBm

### Recordings Management
- Record signals with custom naming and duration
- Batch recording capabilities
- Automatic SNR calculation
- Comprehensive recording metadata storage
- Recording playback functionality
- Export and import capabilities

### Signal Classification
- Machine learning-based signal classification
- Real-time prediction of signal types
- Model training interface with progress tracking
- Save and load trained models
- Feature extraction from recorded signals
- Classification confidence visualization

## Technical Features
- FFT-based spectrum analysis
- Configurable sample rates and gain settings
- Advanced signal processing algorithms
- Neural network-based classification
- JSON-based metadata storage
- Comprehensive logging system

## Requirements

### Hardware
- RTL-SDR compatible USB dongle
- Linux/Windows/MacOS compatible system

### Software Dependencies
```
python >= 3.6
PyQt5
numpy
matplotlib
tensorflow
scipy
pyrtlsdr
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rtlsdr-dev.git
cd rtlsdr-dev
```

2. Create and activate virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Analysis Tab:
   - Use mouse to select frequency regions of interest
   - Scroll to adjust power levels
   - Monitor real-time spectrum and waterfall displays

3. Recording Signals:
   - Select frequency region
   - Configure recording parameters (name, duration, batch size)
   - Start recording
   - Monitor recording progress

4. Signal Classification:
   - Record and label signal samples
   - Train classification model
   - Perform real-time classification
   - Save/load trained models

## Development

The project is structured as follows:

```
rtlsdr-dev/
├── main.py
├── models/
│   └── signal_model.py
├── widgets/
│   ├── classification_spectrum_widget.py
│   └── classification_waterfall_widget.py
├── tabs/
│   └── signal_classification_tab.py
└── recordings/
    └── metadata.json
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

