#(a)Decrease its spatial resolution by half every time and observe its 
# ~change when displaying in the same window size
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ফাংশন: ইমেজের রেজোলিউশন অর্ধেক করা (লুপ ব্যবহার করে)
def decrease_resolution(image):
    height, width = image.shape
    new_height, new_width = height // 2, width // 2
    decreased_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            decreased_image[i, j] = image[i * 2, j * 2]

    return decreased_image

# ধাপ ১: মূল ইমেজ লোড করা এবং রিসাইজ করা
original_image = cv2.imread("image/1a.jpeg", cv2.IMREAD_GRAYSCALE)
original_image = cv2.resize(original_image, (512, 512))  # 512x512 তে রিসাইজ

# ধাপ ২: ধাপে ধাপে রেজোলিউশন কমানো এবং প্লট করা
decreased_image = original_image.copy()
plt.figure(figsize=(20, 20))

for k in range(1, 5):
    plt.subplot(2, 2, k)
    plt.imshow(decreased_image, cmap='gray')
    height, width = decreased_image.shape
    plt.title(f"{height}x{width}")
    decreased_image = decrease_resolution(decreased_image)

plt.tight_layout()
plt.show()
