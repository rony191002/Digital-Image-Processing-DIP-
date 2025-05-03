import cv2
import numpy as np
import matplotlib.pyplot as plt


     
#Function for Harmonic Mean Filter


def apply_harmonic_mean_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape 
    offset, number_of_pixel = mask_size // 2, mask_size * mask_size

    for r in range(height):
        for c in range(width):
            pixel = 0
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        pixel += float(1 / (image[r + x, c + y] + 1e-4))
            pixel = number_of_pixel / pixel
            filtered_image[r, c] = 255 if pixel > 255 else pixel

    return np.uint8(filtered_image)


     
#Function for Geometric Mean Filter

def apply_geometric_mean_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape
    offset = mask_size // 2

    for r in range(height):
        for c in range(width):
            pixel = 1
            count = 0
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        if (image[r + x, c + y]):
                            count += 1
                            pixel = pixel * int(image[r + x, c + y])
            count = 1 if count == 0 else count
            filtered_image[r, c] = pixel ** (1 / count)

    return np.uint8(filtered_image)


     
#Function for adding Salt & Pepper Noise

def add_salt_pepper_noise(image, percent):
    noisy_image = image.copy()
    noise_amount = (image.shape[0] * image.shape[1]) * (percent / 100)

    for k in range(int(noise_amount)):
        index = []
        for x in range(1, 5):
            index.append(np.random.randint(0, image.shape[0]))
        noisy_image[index[0], index[1]], noisy_image[index[2], index[3]] = 0, 255

    return noisy_image


     
#Function for calculating PSNR (Peak Signal to Noise Ratio)

def compute_psnr(image1, image2):
    image1, image2 = np.float64(image1), np.float64(image2)
    mse = np.mean((image1 - image2) ** 2) 
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    
    return round(psnr, 2)

#Loading the Image
character_image = cv2.imread("image/3c.png", 0)
plt.imshow(character_image, cmap = "gray")
plt.title("Original Image")
plt.show()


     

#Adding Salt & Pepper Noise to the Image

noisy_character_image = add_salt_pepper_noise(character_image, 1.5)
plt.imshow(noisy_character_image, cmap = "gray")
plt.title("The Noisy Image")
plt.show()
psnr = compute_psnr(character_image, noisy_character_image)
print(f"PSNR = {psnr}")


#Applying Harmonic Mean Filter

mask_size = 3
harmonic_filtered_image = apply_harmonic_mean_filter(noisy_character_image, mask_size)
plt.imshow(cv2.cvtColor(harmonic_filtered_image, cv2.COLOR_BGR2RGB))
plt.title("After applying Harmonic Mean Filter")
plt.show()
print(f"PSNR = {compute_psnr(noisy_character_image, harmonic_filtered_image)}")


#Applying Geometric Mean Filter

geometric_filtered_image = apply_geometric_mean_filter(noisy_character_image, mask_size)
plt.imshow(cv2.cvtColor(geometric_filtered_image, cv2.COLOR_BGR2RGB))
plt.title("After applying Geometric Mean Filter")
plt.show()
print(f"PSNR = {compute_psnr(noisy_character_image, geometric_filtered_image)}")
