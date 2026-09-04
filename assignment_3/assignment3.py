import cv2
import numpy as np

# Sobel edge detection
def sobel_edge_detection(image):
    img_blur = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=0)
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=1)
    sobel_display = cv2.convertScaleAbs(sobelx)

    cv2.imwrite("solutions/sobel_edge.jpg", sobel_display)
    return sobelx

# Canny edge detection
def canny_edge_detection(image, threshold_1, threshold_2):
    img_blur = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=0)

    canny = cv2.Canny(img_blur, threshold_1, threshold_2)
    cv2.imwrite("solutions/canny_edge.jpg", canny)
    return canny

# Template match
def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9

    result_image = image.copy()
    loc = np.where(res >= threshold)
    h, w = template_gray.shape

    for pt in zip(*loc[::-1]):
        top_left = pt
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 255), 2)

    cv2.imwrite("solutions/temp_match.jpg", result_image)
    return result_image

# Resizing
def resize(image, scale_factor: int, up_or_down: str):
    rows, cols, channels = map(int, image.shape)

    if up_or_down == "up":
        result = cv2.pyrUp(image, dstsize=(cols * scale_factor, rows * scale_factor))
    elif up_or_down == "down":
        result = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
    else:
        return None

    cv2.imwrite(f"solutions/resize_{up_or_down}.jpg", result)
    return result

def main():
    lambo = cv2.imread("lambo.png")  
    shape_img = cv2.imread("shapes-1.png")
    shapes_template = cv2.imread("Shapes_template.jpg")

    sobel_edge_detection(lambo)
    canny_edge_detection(lambo, 50, 50)
    template_match(shape_img, shapes_template)
    resize(lambo, 2, "up")
    resize(lambo, 2, "down")


if __name__ == "__main__":
    main()