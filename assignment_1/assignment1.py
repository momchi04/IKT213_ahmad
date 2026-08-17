import cv2

# Image information
def print_image_information(image):
    img_height, img_width, img_channels = image.shape
    img_size = image.size
    img_data_type = image.dtype

    print("Image Information:")
    print("Height:",img_height)
    print("Width:",img_width)
    print("Channels:",img_channels)
    print("Size:",img_size)
    print("Data type:",img_data_type)

# Camera Information
def save_camera_info():
    cam = cv2.VideoCapture(0)

    cam_fps = cam.get(cv2.CAP_PROP_FPS)
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    with open("solutions/camera_outputs.txt", "w") as file:
        file.write(f"Fps: {cam_fps} \n")
        file.write(f"Height: {cam_height} \n")
        file.write(f"Width: {cam_width} \n")

    cam.release()

def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)

    save_camera_info()

if __name__ == "__main__":
    main()