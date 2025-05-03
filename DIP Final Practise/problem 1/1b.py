import cv2
import numpy as np
import matplotlib.pyplot as plt

# ফাংশন: intensity resolution কমানো
def decrease_resolution(image, bits):
    levels = 2 ** bits                      # কতটি ধাপ থাকবে (যেমন 2-bit → 4 ধাপ)
    step = 255 / (levels - 1)              # প্রতিটি ধাপের মান
    new_image = np.round(image / step) * step  # নতুন ধাপে মান অ্যাডজাস্ট
    return new_image.astype(np.uint8)      # 8-bit unsigned int টাইপে রূপান্তর

# grayscale ছবিটি লোড করি
image = cv2.imread("image/1b.jpg", cv2.IMREAD_GRAYSCALE)

# ছবি দেখানোর জন্য window তৈরি
plt.figure(figsize=(12, 12))

# 8-bit থেকে 1-bit পর্যন্ত ছবিগুলো দেখানো
for i in range(8):
    bits = 8 - i                           # বর্তমান bit-depth (৮ থেকে ১)
    new_img = decrease_resolution(image, bits)

    plt.subplot(2, 4, i+1)                # ২ সারি, ৪ কলাম এর layout
    plt.imshow(new_img, cmap='gray')     # grayscale হিসেবে দেখাই
    plt.title(f"{bits}-bit")             # শিরোনাম


plt.show()
