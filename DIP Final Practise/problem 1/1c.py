import cv2
import numpy as np
import matplotlib.pyplot as plt

# হিস্টোগ্রাম দেখানোর ফাংশন
def show_histogram(image):
    plt.hist(image.ravel(), 256, [0, 256], color='gray')
    plt.title("Histogram")
    plt.xlabel("Gray Level")
    plt.ylabel("Count")
    plt.show()

# 1. ইমেজ লোড করা (গ্রেস্কেল মোডে)
image = cv2.imread('image/1c.jpg', cv2.IMREAD_GRAYSCALE)

# 2. মূল ইমেজ দেখানো
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()

# 3. মূল ইমেজের হিস্টোগ্রাম দেখানো
show_histogram(image)

# 4. থ্রেশোল্ড দিয়ে সেগমেন্ট করা (২৭ এর নিচে হলে ০, নয় তো ২৫৫)
threshold = 27
segmented_image = np.where(image < threshold, 0, 255).astype(np.uint8)

# 5. ডেটাটাইপ কনভার্ট করা (uint8 = 0 থেকে 255 পর্যন্ত ভ্যালু রাখার জন্য)

# 6. সেগমেন্টেড ইমেজ দেখানো
plt.imshow(segmented_image, cmap='gray')
plt.title("Segmented Image")
plt.show()

# 7. সেগমেন্টেড ইমেজের হিস্টোগ্রাম দেখানো
show_histogram(segmented_image)
