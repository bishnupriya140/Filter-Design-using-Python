import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz,lfilter
# Filter parameters
M = 51 # Number of taps
low_cutoff_freq = 0.2 # Low cutoff frequency (normalized frequency, 0.5 corresponds to Nyquist

high_cutoff_freq = 0.4 
sampling_frequency = 10000  

# Design the FIR filter using the window method with a bandpass configuration
taps = firwin(M, [low_cutoff_freq, high_cutoff_freq], pass_zero=False,window='hamming')

# Plot the frequency response
w, h = freqz(taps, worN=8000)
fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response BPF')
ax1.plot(0.5 * w / np.pi, np.abs(h), 'b')
ax1.set_ylabel('Amplitude', color='b')
ax1.set_xlabel('Frequency [cycles/sample]')
ax1.grid()
plt.show()

# Frequency axis
frequency_hz = w * sampling_frequency / (2 * np.pi)

# Plot magnitude response
plt.figure()
plt.plot(frequency_hz, 20 * np.log10(abs(h)))
plt.title('Frequency Response of Band-pass Filter (Hamming Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()

# Calculate phase response
phase_response = np.unwrap(np.angle(h))
# Frequency axis
frequency_hz = w * sampling_frequency / (2 * np.pi)

# Plot phase response
plt.subplot()
plt.plot(frequency_hz, phase_response)
plt.title('Phase Response of Band-pass Filter (Hamming Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.tight_layout()
plt.show()


# Generate impulse signal
impulse = np.zeros(M)
impulse[0] = 1

# Compute the impulse response of the filter
impulse_response = lfilter(taps, 1, impulse)

# Time axis
t = np.arange(M) / sampling_frequency

# Plot impulse response
plt.figure()
plt.plot(t, impulse_response)
plt.title('Impulse Response of Band-pass Filter (Hamming Window)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Calculate the time-domain representation using the inverse Fourier transform
time_domain_response = np.fft.ifft(taps)

# Time axis
t = np.arange(len(time_domain_response)) / sampling_frequency

# Plot truncated time-domain response
plt.figure()
plt.plot(t, np.real(time_domain_response))
plt.title('Truncated Time-domain Response of Band-pass Filter (Hamming Window)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


