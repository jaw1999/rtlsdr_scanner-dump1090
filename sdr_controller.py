# sdr_controller.py
from PyQt5.QtCore import QObject, pyqtSignal
from rtlsdr import RtlSdr
import numpy as np
import threading

class SDRController(QObject):
    # Define PyQt5 signals
    dump1090_started = pyqtSignal()
    dump1090_stopped = pyqtSignal()

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SDRController, cls).__new__(cls)
                cls._instance.init()
            return cls._instance

    def init(self):
        self.sdr = None
        self.center_freq = 100e6  # 100 MHz
        self.sample_rate = 2.4e6  # 2.4 MHz
        self.gain = 'auto'
        self.fft_size = 1024
        self.averaging = 1
        self.num_samples = 256 * 1024
        self.is_active = False  # Indicates if SDR is currently in use

    def initialize(self):
        if self.is_active:
            print("SDR is currently in use. Cannot initialize.")
            return False
        try:
            if self.sdr is not None:
                self.close()
            self.sdr = RtlSdr()
            return True
        except Exception as e:
            print(f"Error initializing SDR: {str(e)}")
            return False

    def setup(self):
        if self.sdr is None:
            if not self.initialize():
                return False
        try:
            self.sdr.center_freq = self.center_freq
            self.sdr.sample_rate = self.sample_rate
            if self.gain == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(self.gain)
            self.is_active = True
            return True
        except Exception as e:
            print(f"Error setting up SDR: {str(e)}")
            return False

    def read_samples(self):
        if self.sdr is None:
            print("SDR not initialized")
            return None
        try:
            return self.sdr.read_samples(self.num_samples)
        except Exception as e:
            print(f"Error reading samples: {str(e)}")
            return None

    def compute_power_spectrum(self, samples):
        if samples is None:
            return None, None

        try:
            averaged_spectrum = np.zeros(self.fft_size)
            for i in range(self.averaging):
                start = i * self.fft_size
                end = start + self.fft_size
                chunk = samples[start:end]
                if len(chunk) < self.fft_size:
                    break  # Avoid incomplete chunks
                spectrum = np.abs(np.fft.fftshift(np.fft.fft(chunk, n=self.fft_size))) ** 2
                averaged_spectrum += spectrum
            averaged_spectrum /= self.averaging

            power_spectrum_db = 10 * np.log10(averaged_spectrum + 1e-12)  # Added epsilon to avoid log(0)
            freq_range = np.linspace(self.center_freq - self.sample_rate/2,
                                     self.center_freq + self.sample_rate/2,
                                     len(power_spectrum_db)) / 1e6
            return freq_range, power_spectrum_db
        except Exception as e:
            print(f"Error computing power spectrum: {str(e)}")
            return None, None

    def get_available_gains(self):
        if self.is_active:
            print("SDR is currently in use. Cannot retrieve gains.")
            return []
        if self.sdr is None:
            if not self.initialize():
                return []
        try:
            gains = self.sdr.get_gains()
            # Filter out unreasonable values and convert to tenths of dB
            return [f"{gain:.1f}" for gain in gains if 0 <= gain <= 50]
        except Exception as e:
            print(f"Error getting available gains: {str(e)}")
            return []

    def set_frequency_correction(self, ppm):
        if self.sdr is None:
            print("SDR not initialized")
            return False
        try:
            self.sdr.freq_correction = int(ppm)
            return True
        except Exception as e:
            print(f"Error setting frequency correction: {str(e)}")
            return False

    def close(self):
        if self.sdr is not None:
            try:
                self.sdr.close()
                self.sdr = None
                self.is_active = False
            except Exception as e:
                print(f"Error closing SDR: {str(e)}")

    def reset(self):
        self.close()
        return self.initialize()
