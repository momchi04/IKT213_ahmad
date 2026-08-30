import cv2
import numpy as np

# 1. Padding
def padding(image, border_width):
    image_padded = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/iris-1_padded.jpg", image_padded)
    return image_padded

# 2. Cropping
def crop(image, x_0, x_1, y_0, y_1):
    image_cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("solutions/iris-1_cropped.jpg", image_cropped)
    return image_cropped

# 3. Resize
def resize(image, width, height):
    image_resized = cv2.resize(image, (width, height))
    cv2.imwrite("solutions/iris-1_resized.jpg", image_resized)
    return image_resized

# 4. Manual copy
def copy(image, emptyPictureArray):
    img_height, img_width, img_channels = image.shape
    for y in range (0, img_height):
        for x in range (0, img_width):
            emptyPictureArray[y, x, :] = image[y, x, :]
    cv2.imwrite("solutions/iris-1_copy.jpg", emptyPictureArray)
    return emptyPictureArray

# 5. Grayscale
def grayscale(image):
    image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/iris-1_grayscaled.jpg", image_grayscaled)
    return image_grayscaled

# 6. HSV
def hsv(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/iris-1_hsv.jpg", image_hsv)
    return image_hsv

# 7. Color shifting
def hue_shifted(image, emptyPictureArray, hue):
    img_height, img_width, img_channels = emptyPictureArray.shape
    for y in range (0, img_height):
        for x in range (0, img_width):
            emptyPictureArray[y, x, :] = (image[y, x, :]) + hue
    cv2.imwrite("solutions/iris-1_hue_shifted.jpg", emptyPictureArray)
    return emptyPictureArray

# 8. Smoothing
def smoothing(image):
    image_smoothed = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite("solutions/iris-1_smoothed.jpg", image_smoothed)
    return image_smoothed

# 9. Rotation
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        image_rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        print("Rotated image 90 degrees, but not saving it, only saving 180 degree rotation")
    if rotation_angle == 180:
        image_rotate = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("solutions/iris-1_180_degrees.jpg", image_rotate)
    return image_rotate


def main():
    image = cv2.imread("iris-1.jpg")
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    padding(image, 100)
    crop(image, 200, -130, 200, -130)
    resize(image, 200, 200)
    copy(image, emptyPictureArray)
    grayscale(image)
    hsv(image)
    hue_shifted(image, emptyPictureArray, 50)
    smoothing(image)
    rotation(image, 90)
    rotation(image, 180)


if __name__ == "__main__":
    main()