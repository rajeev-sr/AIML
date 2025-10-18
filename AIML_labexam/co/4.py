import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -------------------------------
# 1. Load Audio
# -------------------------------
audio_path = librosa.example('trumpet')  # sample audio provided by librosa
y, sr = librosa.load(audio_path)

print(f"Audio loaded: {audio_path}")
print(f"Duration: {len(y)/sr:.2f} seconds, Sampling Rate: {sr}")

# -------------------------------
# 2. Convert to Mel Spectrogram
# -------------------------------
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Convert Spectrogram → Image (for segmentation with OpenCV)
# -------------------------------
# Normalize spectrogram values to 0–255
spectrogram_img = cv2.normalize(S_dB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply thresholding to segment "high-energy" regions
_, thresh_img = cv2.threshold(spectrogram_img, 150, 255, cv2.THRESH_BINARY)

# -------------------------------
# 4. Contour Detection
# -------------------------------
contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

segmented_img = cv2.cvtColor(spectrogram_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 2)

# -------------------------------
# 5. Show Original vs Segmented
# -------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original Spectrogram")
plt.imshow(spectrogram_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Spectrogram")
plt.imshow(segmented_img)
plt.axis("off")

plt.show()

