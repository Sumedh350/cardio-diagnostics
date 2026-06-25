import serial
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# CHANGE THIS
PORT = 'COM3'

# Open serial connection
ser = serial.Serial(PORT, 115200)

print("Recording started...")

samples = []

# Collect 8000 samples (~4 sec at 2kHz)
while len(samples) < 8000:
    try:
        line = ser.readline().decode().strip()

        if line:
            value = int(line)
            samples.append(value)

    except:
        pass

ser.close()

print("Recording complete")

# Convert to numpy array
audio = np.array(samples)

# Remove DC offset
audio = audio - np.mean(audio)

# Normalize
audio = audio / np.max(np.abs(audio))

# Convert to int16
audio_int16 = (audio * 32767).astype(np.int16)

# Save WAV
write("heart_sound.wav", 2000, audio_int16)

print("Audio saved as heart_sound.wav")

# Plot waveform
plt.figure(figsize=(12,4))
plt.plot(audio)
plt.title("Heart Sound Waveform")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()