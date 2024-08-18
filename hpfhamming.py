import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Parameters
cutoff_frequency = 0.1# Cutoff frequency in Hz
sampling_frequency = 10000  # Sampling frequency in Hz
filter_order = 51  # Filter order

# Low-pass filter design using Hamming window
b = signal.firwin(filter_order, cutoff=cutoff_frequency/(sampling_frequency/2), window='hamming')

# Spectral inversion to convert low-pass filter to high-pass
b_hp = -b
b_hp[filter_order // 2] += 1

# Apply the filter to a test signal
t = np.linspace(0, 1, sampling_frequency)
x = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 2000 * t)  # Test signal with 500 Hz and 2000 Hz components
y = signal.lfilter(b_hp, 1, x)

# Plot the frequency response
w, h = signal.freqz(b_hp)
plt.figure()
plt.title('Frequency response')
plt.plot(w, 20 * np.log10(abs(h)))
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [rad/sample]')
plt.grid(which='both', axis='both')
plt.show()

# Plot the original and filtered signals
plt.figure()
plt.plot(t, x, label='Original Signal')
plt.plot(t, y, label='Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Original and Filtered Signals')
plt.legend()
plt.grid(True)
plt.show()

# Calculate phase response
phase_response = np.angle(h)
# Convert frequency from radians/sample to Hz
frequency_hz = w * sampling_frequency / (2 * np.pi)
# Plot phase response
plt.figure()
plt.plot(frequency_hz, phase_response)
plt.title('Phase Response of High-pass Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.show()

# Create an impulse signal
impulse = np.zeros(filter_order)
impulse[0] = 1

# Filter the impulse signal to get the impulse response
impulse_response = signal.lfilter(b_hp, 1, impulse)

# Time axis
t = np.arange(filter_order) / sampling_frequency

# Plot impulse response
plt.figure()
plt.plot(t, impulse_response)
plt.title('Impulse Response of High-pass Filter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# Calculate frequency axis
frequency_hz = w * sampling_frequency / (2 * np.pi)
# Calculate the IDTFT
idtft = np.fft.ifft(h, n=filter_order)
# Time axis
t = np.arange(filter_order) / sampling_frequency
# Plot IDTFT
plt.figure()
plt.plot(t, np.real(idtft))
plt.title('Inverse Discrete-Time Fourier Transform (IDTFT) of High-pass Filter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Parameters
cutoff_frequency = 1000  # Cutoff frequency in Hz
sampling_frequency = 10000  # Sampling frequency in Hz
filter_order = 51  # Filter order

# High-pass filter design using Hamming window
b = signal.firwin(filter_order, cutoff=cutoff_frequency/(sampling_frequency/2),pass_zero=False, window='hamming')

# Compute the frequency response of the filter
w, h = signal.freqz(b)

# Convert frequency from radians/sample to Hz
frequency_hz = w * sampling_frequency / (2 * np.pi)

# Plot magnitude response
plt.figure()
plt.plot(frequency_hz, 20 * np.log10(abs(h)))
plt.title('Frequency Response of High-pass Filter (Hamming Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()

