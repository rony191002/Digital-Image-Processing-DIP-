#(b)Observe the ringing effect of ideal low pass filter on the image. Use different radius (D0) of ideal low pass filter and display their results
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Ideal_Law_Pass_Filter
def apply_ideal_low_pass_filter(fft_image, cutoff):
    h, w = fft_image.shape
    y, x = np.ogrid[:h, :w]  # Efficient grid creation
    D = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)

    H = np.zeros_like(fft_image)  # Ideal Low pass formula
    H[D <= cutoff] = 1 
    filtered_image = fft_image * H
    result = np.fft.ifft2(np.fft.ifftshift(filtered_image))
    return np.abs(result)

# -------------------------------
# Load Grayscale Image
# -------------------------------
image = cv2.imread("image/4b.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()


# -------------------------------
# Add Gaussian Noise
# -------------------------------
mean, std_dev = 7, 13
gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)

noisy_image = image.astype(np.float32) + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

plt.imshow(noisy_image, cmap='gray')
plt.title("Image with Gaussian Noise")
plt.show()


# -------------------------------
# Apply FFT to Noisy Image
# -------------------------------
fft_image = np.fft.fftshift(np.fft.fft2(noisy_image))

# -------------------------------
# Apply Ideal Low Pass Filters of Various Radius
# -------------------------------
cutoff_values = [5, 10, 20, 30, 40, 50]  # বিভিন্ন cutoff frequency

plt.figure(figsize=(12, 8))

for D0 in cutoff_values:
    ideal_img = apply_ideal_low_pass_filter(fft_image, cutoff=D0)
    plt.imshow(ideal_img, cmap='gray')
    plt.title(f"Ideal Low Pass Filter (Cutoff = {D0})")
    plt.axis('off')
    plt.show()

    
