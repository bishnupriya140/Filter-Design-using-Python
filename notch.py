import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Filter specifications
fs = 500              # Sampling frequency in Hz
f_center = 100        # Center frequency in Hz
bandwidth = 20        # Bandwidth in Hz

# Calculate normalized cutoff frequencies
nyquist_freq = 0.5 * fs
normalized_f_center = f_center / nyquist_freq
normalized_bandwidth = bandwidth / nyquist_freq
# Determine the number of taps (making it odd)
num_taps = 2 * int(np.ceil(8 * fs / bandwidth)) + 1
# Design the FIR filter using the Blackman window method
taps = signal.firwin(num_taps, [normalized_f_center - 0.5 * normalized_bandwidth, 
                                normalized_f_center + 0.5 * normalized_bandwidth], 
                      window='blackman')
# Compute the frequency response
w, h = signal.freqz(taps, worN=8000)
# Plot the frequency response
plt.figure()
plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(np.abs(h)), 'b')
plt.title('Frequency Response of Notch FIR Filter using Blackman Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()
# Compute the frequency response
w, h = signal.freqz(taps, worN=8000)
# Compute phase response in radians
phase_response_rad = np.angle(h)
# Plot the phase response
plt.figure()
plt.plot(0.5 * fs * w / np.pi, phase_response_rad, 'b')
plt.title('Phase Response of Narrow-Band FIR Filter using Blackman Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.show()

