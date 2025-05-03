# 3.(b) Use different sizes of mask (3×3, 5×5, 7×7) with average filter for noise suppression
# and observe their performance in terms of PSNR.
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Function for Average Spatial Filter

def average_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape
    offset, weight = mask_size // 2, mask_size * mask_size

    for r in range(height):
        for c in range(width):
            filtered_image[r, c] = 0;
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        filtered_image[r, c] += (image[r + x, c + y] / weight)
    
    return np.uint8(filtered_image)

#Function for adding Salt & Pepper Noise
def add_salt_and_pepper_noise(image, percent):
    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    noise_amount = int(total_pixels * (percent / 100))

    for _ in range(noise_amount):
        row = np.random.randint(0, image.shape[0])
        col = np.random.randint(0, image.shape[1])

        # Randomly choose between salt or pepper
        if np.random.rand() < 0.5:
            noisy_image[row, col] = 0     # pepper
        else:
            noisy_image[row, col] = 255   # salt

    return noisy_image

#Function for calculating PSNR (Peak Signal to Noise Ratio)
def compute_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2) 
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    return round(psnr,2)

#Loading the Image
character_image = cv2.imread("image/3b.png", 0)
plt.imshow(character_image, cmap = "gray")
plt.title("Original Image")
plt.show()

#Adding Salt & Pepper Noise to the Image
noisy_character_image = add_salt_and_pepper_noise(character_image, 15)
plt.imshow(noisy_character_image, cmap = "gray")
plt.title("The Noisy Image")
plt.show()
psnr = compute_psnr(character_image, noisy_character_image)
print(f"PSNR = {psnr}")

#Applying average filter with (3x3, 5x5, 7x7) mask
for mask_size in range(3, 8, 2):
    avg_character_image = average_filter(noisy_character_image, mask_size)
    plt.imshow(cv2.cvtColor(avg_character_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Noisy Image after applying Averaging Filter of mask size {mask_size}x{mask_size}")
    plt.show()
    print(f"PSNR = {compute_psnr(noisy_character_image, avg_character_image)}")
