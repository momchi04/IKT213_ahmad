import cv2
import numpy as np

# Harris Corner Detection
def harris_corner_detection(reference_image):
    marked = reference_image.copy()
    gray = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    marked[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite("solutions/harris.jpg", marked)
    return marked

# Feature-Based Image Alignment
# Using SIFT:
def sift_image_allignemnet(image_to_align, reference_image, max_features, good_match_precent):
    MIN_MATCH_COUNT = 10

    align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(align_gray, None)
    kp2, des2 = sift.detectAndCompute(ref_gray, None)
    print("keypoints in align image:", len(kp1))
    print("keypoints in reference image:", len(kp2))

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_precent*n.distance:
            good.append(m)
    print("good matches after ratio test:", len(good))

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h_ref, w_ref = ref_gray.shape
        aligned = cv2.warpPerspective(align_gray, M, (w_ref, h_ref))

        draw_params = dict(matchColor=(0,255,0), singlePointColor=None,
                            matchesMask=matchesMask, flags=2)
        matches_img = cv2.drawMatches(align_gray, kp1, ref_gray, kp2, good, None, **draw_params)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        aligned = None
        matches_img = None

    cv2.imwrite("solutions/aligned.jpg", aligned)
    cv2.imwrite("solutions/matches.jpg", matches_img)
    return aligned, matches_img
    

def main():
    reference_image = cv2.imread("reference_img.png")
    image_to_align = cv2.imread("align_this.jpg")
    harris_corner_detection(reference_image)
    sift_image_allignemnet(image_to_align, reference_image, 10, 0.7)


if __name__ == "__main__":
    main()