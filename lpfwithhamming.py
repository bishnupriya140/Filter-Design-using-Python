import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, firwin,hamming,lfilter
from scipy.fft import ifft

window = np.hamming(51) 
plt.plot(window) 
plt.title("Hamming Window") 
plt.ylabel("Amplitude") 
plt.xlabel("Sample") 
plt.show() 

# Filter parameters
num_taps = 51
cutoff_freq_lp = 0.1
nyquist_rate = 2000 

# Design the FIR filters
fir_coeffs_lp = firwin(num_taps, cutoff_freq_lp,window='hamming')

# Frequency response

w_lp, h_lp = freqz(fir_coeffs_lp)
freq = w_lp * nyquist_rate / np.pi / 2

# Plot the frequency responses
plt.figure(figsize=(10, 8))

#plt.subplot(2, 2, 1)
plt.plot(0.5 * np.pi * w_lp, np.abs(h_lp), 'b')
plt.title('Low-pass FIR Filter')
plt.xlabel('Frequency [radians / sample]')
plt.ylabel('Magnitude')
plt.grid()

taps_hamming = hamming(num_taps)

# Impulse response
impulse_response = np.zeros(num_taps)
impulse_response[num_taps // 2] = 1  # Dirac delta at the center
impulse_response_hamming = impulse_response * taps_hamming

# Frequency response
w_hamming, h_hamming = freqz(impulse_response_hamming, worN=8000)
freq_hamming = w_hamming * nyquist_rate / np.pi / 2

# Plot impulse response
plt.figure(figsize=(10, 6))
plt.stem(impulse_response_hamming, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title('Hamming Window Impulse Response')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid()
plt.show()


# Plot frequency response
plt.figure(figsize=(10, 6))
#plt.subplot(2, 2, 3)
plt.plot(freq_hamming, 20 * np.log10(abs(h_hamming)), 'r')
plt.title('Hamming Window Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()
plt.show()

# Plot phase response
plt.figure(figsize=(10, 6))
#plt.subplot(2, 2, 4)
plt.plot(freq_hamming, np.unwrap(np.angle(h_hamming)), 'b')
plt.title('Hamming Window Phase Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.show()

#plot Frequency response
plt.figure(figsize=(10, 6))
plt.plot(freq, 20 * np.log10(abs(h_lp)), 'b')
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()

#plot Impulse response with Hamming window
impulse_response = ifft(h_lp)
# Plot impulse response
plt.figure(figsize=(10, 6))
plt.stem(np.real(impulse_response), linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title('Impulse Response of Low-Pass Filter with Hamming Window (IDTFT)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Generate a sample signal
t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector
f_signal = 20  # Frequency of the signal in Hz
sample_signal = np.sin(2 * np.pi * f_signal * t)  # Sample signal

# Filter the sample signal
filtered_signal_lpf = lfilter(fir_coeffs_lp, 1, sample_signal)

# Plot the time-domain sequence
plt.figure(figsize=(10, 6))
plt.plot(t, sample_signal, label='Original Signal')
plt.plot(t, filtered_signal_lpf, label='LPF Filtered Signal (Hamming Window)')
plt.title('Time Domain Sequence of LPF Filtered Signal (Hamming Window)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

plt.tight_layout()
plt.show()