#(a)	Apply 4th order Butterworth and Gaussian low pass filter to analyze their performance quantitatively

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Butterworth Low Pass Filter using ogrid
# -------------------------------
def apply_butterworth_filter(fft_image, order, cutoff):
    h, w = fft_image.shape
    y, x = np.ogrid[:h, :w]  # Efficient grid creation
    D = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)

    H = 1 / (1 + (D / cutoff) ** (2 * order))  # Butterworth formula
    filtered_image = fft_image * H
    result = np.fft.ifft2(np.fft.ifftshift(filtered_image))
    return np.abs(result)

# -------------------------------
# Gaussian Low Pass Filter using ogrid
# -------------------------------
def apply_gaussian_filter(fft_image, cutoff):
    h, w = fft_image.shape
    y, x = np.ogrid[:h, :w]  # Efficient grid creation
    D = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)

    H = np.exp(-(D ** 2) / (2 * (cutoff ** 2)))  # Gaussian formula
    filtered_image = fft_image * H
    result = np.fft.ifft2(np.fft.ifftshift(filtered_image))
    return np.abs(result)

# -------------------------------
# Load Grayscale Image
# -------------------------------
image = cv2.imread("image/4a.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()

# -------------------------------
# Add Gaussian Noise
# -------------------------------
mean, std_dev = 7, 13
gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)

#এখানে mean হলো গাউসিয়ান নয়েজের গড় (মাঝের মান), আর std_dev হলো standard deviation, মানে নয়েজের ছড়িয়ে থাকার পরিমাণ।
#এই ক্ষেত্রে গড় = ৭ এবং standard deviation = ১৩।
#এই লাইনে np.random.normal() ব্যবহার করে image.shape অনুযায়ী (অর্থাৎ ইমেজের আকার অনুযায়ী) গাউসিয়ান নয়েজ তৈরি করা হয়েছে।
# এই নয়েজ হবে float32 টাইপে কনভার্ট করা, যাতে গাণিতিক অপারেশন ঠিকমতো হয়।

noisy_image = image.astype(np.float32) + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

#এখানে মূল ইমেজকে float32 টাইপে কনভার্ট করে গাউসিয়ান নয়েজ যোগ করা হয়েছে।
#ইমেজে নয়েজ যুক্ত হচ্ছে এখন।
#np.clip() ফাংশন ব্যবহার করে নিশ্চিত করা হয়েছে যে ইমেজের সব পিক্সেল ভ্যালু 0 থেকে 255-এর মধ্যে থাকে।


plt.imshow(noisy_image, cmap='gray')
plt.title("Image with Gaussian Noise")
plt.show()

# -------------------------------
# Apply FFT to Noisy Image
# -------------------------------
fft_image = np.fft.fftshift(np.fft.fft2(noisy_image))

# -------------------------------
# Apply Butterworth Filter
# -------------------------------
butter_img = apply_butterworth_filter(fft_image, order=2, cutoff=25)
plt.imshow(butter_img, cmap='gray')
plt.title("After Butterworth Filter")
plt.show()

# -------------------------------
# Apply Gaussian Filter
# -------------------------------
gaussian_img = apply_gaussian_filter(fft_image, cutoff=25)
plt.imshow(gaussian_img, cmap='gray')
plt.title("After Gaussian Filter")
plt.show()
