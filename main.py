import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

# List to store the brightness values of all images
all_y_values = []


# Show a brightness chart for all images after processing
def show_overall_brightness_histogram():
    all_y = np.concatenate(all_y_values)
    plt.figure(figsize=(10, 5))
    plt.hist(all_y, bins=256, range=(0, 256), color='gray')
    plt.title("Overall Brightness Histogram (Y Channel)")
    plt.xlabel("Brightness (0-255)")
    plt.ylabel("Pixel Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("overall_brightness_histogram.png")
    plt.show()
    print("Saved brightness chart as 'overall_brightness_histogram.png'")


# This checks how bright the image is on average
def compute_brightness(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]  # Y is the brightness part
    return np.mean(y_channel)  # Find the average brightness


def auto_gamma_correction(image, target_brightness=128):
    brightness = compute_brightness(image)

    gamma = target_brightness / brightness
    gamma = max(0.9, min(1.5, gamma))
    # we need to cap it to make sure the image is not over adjusted
    normalized = image / 255.0
    corrected = np.power(normalized, 1 / gamma)  # Adjust brightness
    output = np.uint8(corrected * 255)
    return output


# This fills in missing or broken parts of the image
def improved_inpainting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)


# This fixes weird lighting by balancing the colors
def gray_world_white_balance(image):
    result = image.astype(np.float32)

    avgB = np.mean(result[:, :, 0])
    avgG = np.mean(result[:, :, 1])
    avgR = np.mean(result[:, :, 2])
    avg_gray = (avgB + avgG + avgR) / 3

    result[:, :, 0] *= (avg_gray / avgB)
    result[:, :, 1] *= (avg_gray / avgG)
    result[:, :, 2] *= (avg_gray / avgR)

    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def noise(image):
    # removing salt-and-pepper noise
    image = cv2.medianBlur(image, 3)
    # removing gaussian noise
    image = cv2.bilateralFilter(image, d=9, sigmaColor=40, sigmaSpace=40)
    return image


def correct_warping(image):
    rows, cols = image.shape[:2]
    # fixed pt since all photo warping the same way
    src_points = np.float32([
        [9, 14],
        [235, 3],
        [31, 242],
        [251, 236]
    ])
    dst_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (cols, rows))
    # this function will keep the same resolution
    return warped


# This makes the image look sharper and clearer
def sharpen(image, strength=1.5, blur_size=(5, 5)):
    blurred = cv2.GaussianBlur(image, blur_size, 0)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:  # error handling
        print("Error: Could not load image")
        return None, None

    original = image.copy()

    # The order of these steps matters (will affect accuracy of the classifier)
    image = improved_inpainting(image)
    image = correct_warping(image)
    image = sharpen(image)
    image = noise(image)
    image = auto_gamma_correction(image)
    image = gray_world_white_balance(image)

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    all_y_values.append(y_channel.flatten())

    return original, image


# Process all images in a folder
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)

    for img_name in sorted(os.listdir(input_dir)):
        # eror handling
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, img_name)
        _, processed_img = process_image(img_path)

        if processed_img is not None:
            save_path = os.path.join(output_dir, img_name)
            cv2.imwrite(save_path, processed_img)
            # this process is made since i have try nlm before
            # and it takes time so print the process
            print(f"Processed: {img_name}")


if __name__ == "__main__":
    main()
    # If you want to see the brightness chart, uncomment this:
    # show_overall_brightness_histogram()
