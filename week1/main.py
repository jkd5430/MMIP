# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:43:00 2026

@author: u1021
"""
import os
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
# %%
# 第一題
img_path = r"C:\Users\u1021\Downloads\MMIP\week1\data\00d068ac58deb9599a557d85bf024065.jpg"
img = cv2.imread(img_path)
numpy_times = []
opencv_times = []
for i in range(100):

    start = time.perf_counter()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #cv算法
    
    end = time.perf_counter()
    opencv_times.append(end - start)

    start = time.perf_counter()

    Gray = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]  #np算法
    Gray = Gray.astype(np.uint8)
    # Gray = np.round(Gray).astype(np.uint8) #四捨五入
    
    end = time.perf_counter()
    numpy_times.append(end - start)


numpy_avg = np.mean(numpy_times)
opencv_avg = np.mean(opencv_times)
print(numpy_avg)
print("np")
print(opencv_avg)
print("cv")
print(np.sum(Gray-img_gray))
# cv2.imshow('og Image', img)
# cv2.imshow('cv gray Image', img_gray)
# cv2.imshow('my gray Image', Gray)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# %%
# 第二題 he


img_path = r"C:\Users\u1021\Downloads\MMIP\week1\data\00d068ac58deb9599a557d85bf024065.jpg"
img = cv2.imread(img_path,0)
# plt.hist(img.ravel(), 256, [0, 255])
# plt.savefig('C:/Users/u1021/Downloads/input_image_grayscale.png', bbox_inches='tight', dpi=300)
def he(img,bit):
    b=np.reshape(img, img.shape[0]*img.shape[1])
    bit=2**bit
    ni=[0]*bit
    pdf=[0]*bit
    cdf=[0]*bit
    fi=[0]*bit
    new_intensity=[0]*bit
    new_ni=[0]*bit
    for i in range(len(b)):
        ni[b[i]]=ni[b[i]]+1
        
    for j in range(len(pdf)):
        pdf[j]=ni[j]/float(img.shape[0]*img.shape[1])
        cdf[j]=sum(pdf)
        fi[j]=sum(pdf)*(bit-1)
        new_intensity[j]=round(sum(pdf)*(bit-1))
        new_ni[new_intensity[j]]=new_ni[new_intensity[j]]+ni[j]
    # print(pdf)
    # print(cdf)
    # print(fi)
    
    equalized_img = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            old_value = img[y, x]
            equalized_img[y, x] = new_intensity[old_value]

    return equalized_img, new_ni
# plt.hist(equalize_img.ravel(), 256, [0, 255])

numpy_times = []
opencv_times = []

for i in range(100):

    start = time.perf_counter()
    equalize_img = cv2.equalizeHist(img)

    end = time.perf_counter()
    opencv_times.append(end - start)
    
    start = time.perf_counter()
    np_he_img,np_he=he(img,8)

    end = time.perf_counter()
    numpy_times.append(end - start)
numpy_avg = np.mean(numpy_times)
opencv_avg = np.mean(opencv_times)
print(np.sum(np_he_img-equalize_img))
print(numpy_avg)
print("np_Time")
print(opencv_avg)
print("cv_Time")
# plt.figure(figsize=(12,6))
# plt.subplot(2,3,1)
# plt.imshow(img, cmap = 'gray')
# plt.title('og image ')

# plt.subplot(2,3,2)
# plt.imshow(equalize_img, cmap = 'gray')
# plt.title('cv_he image')

# plt.subplot(2,3,3)
# plt.imshow(np_he_img, cmap = 'gray')
# plt.title('np_he image')

# plt.subplot(2,3,4)
# plt.hist(img.ravel(), 256, [0, 255])
# plt.title('og image hit')

# plt.subplot(2,3,5)
# plt.hist(equalize_img.ravel(), 256, [0, 255])
# plt.title('cv_he image hit ')

# plt.subplot(2,3,6)
# plt.hist(np_he_img.ravel(), 256, [0, 255])
# plt.title('np_he image hit ')

# plt.savefig('C:/Users/u1021/Downloads/MMIP/week1/data/all.png', bbox_inches='tight', dpi=300)

# %%
#題目三 梯形轉換 需要自己點
img_path = r"C:\Users\u1021\Downloads\MMIP\week1\data\IMG_20200204_112217.jpg"
img = cv2.imread(img_path,0)
points = []
img = cv2.resize(img, (600, 600))

def mouse_callback(event, x, y, flags, param):  # 順序 左上 右上 右下 左下
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")

            # 畫點方便確認
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img)

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
# 轉成 OpenCV 需要的格式
src_pts = np.float32(points)
width = 600
height = 900

dst_pts = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

result = cv2.warpPerspective(
    img,
    M,
    (width, height)
)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#題目三 梯形轉換 自動化
img_path = r"C:\Users\u1021\Downloads\MMIP\week1\data\20260921_171137.jpg"
img = cv2.imread(img_path,0)
img = cv2.equalizeHist(img)

width = 500
height = 500
threshold_value, binary = cv2.threshold(
    img,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary,
    connectivity=8
)
largest_label = 1
largest_area = stats[1, cv2.CC_STAT_AREA]

for i in range(2, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    if area > largest_area:
        largest_area = area
        largest_label = i

print("最大連通區域 label =", largest_label)
print("最大連通區域面積 =", largest_area)

largest_mask = np.zeros_like(binary)
largest_mask[labels == largest_label] = 255

ys, xs = np.where(labels == largest_label)

# 轉成 [x, y]
points = np.column_stack((xs, ys))

# x + y
s = points[:, 0] + points[:, 1]

# x - y
diff = points[:, 0] - points[:, 1]

# 左上：x + y 最小
top_left = points[np.argmin(s)]

# 右下：x + y 最大
bottom_right = points[np.argmax(s)]

# 右上：x - y 最大
top_right = points[np.argmax(diff)]

# 左下：x - y 最小
bottom_left = points[np.argmin(diff)]

# 固定順序：
# 左上、右上、右下、左下
src_pts = np.float32([
    top_left,
    top_right,
    bottom_right,
    bottom_left
])
show_img = img.copy()

for point in src_pts:
    px = int(point[0])
    py = int(point[1])

    cv2.circle(show_img, (px, py), 8, (0, 0, 255), -1)

dst_pts = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

result = cv2.warpPerspective(
    img,
    M,
    (width, height)
)
showimg = cv2.resize(img, (500, 500))
showbinary = cv2.resize(binary, (500, 500))
largest_mask = cv2.resize(largest_mask, (500, 500))

cv2.imshow("Original", showimg)
cv2.imshow("Otsu", showbinary)
cv2.imshow("result", result)
cv2.imshow("largest_mask", largest_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#第四題


# ============================================================
# Feather blending
# ============================================================
def feather_blend(img1, img2, mask1, mask2):

    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)

    # 計算每個 pixel 到邊界的距離
    dist1 = cv2.distanceTransform(
        m1,
        cv2.DIST_L2,
        5
    )

    dist2 = cv2.distanceTransform(
        m2,
        cv2.DIST_L2,
        5
    )

    only1 = (m1 == 1) & (m2 == 0)
    only2 = (m1 == 0) & (m2 == 1)
    overlap = (m1 == 1) & (m2 == 1)

    weight1 = np.zeros_like(dist1, dtype=np.float32)
    weight2 = np.zeros_like(dist2, dtype=np.float32)

    # 非重疊區直接使用原圖
    weight1[only1] = 1.0
    weight2[only2] = 1.0

    # 重疊區根據距離做 blending
    total_dist = dist1 + dist2 + 1e-6

    weight1[overlap] = (
        dist1[overlap] /
        total_dist[overlap]
    )

    weight2[overlap] = (
        dist2[overlap] /
        total_dist[overlap]
    )

    # 增加 channel 維度
    weight1 = weight1[:, :, None]
    weight2 = weight2[:, :, None]

    result = (
        img1.astype(np.float32) * weight1 +
        img2.astype(np.float32) * weight2
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return result


# ============================================================
# 自動裁掉 panorama 外圍黑色區域
# ============================================================
def crop_panorama(img, mask):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return img

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    return img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]


# ============================================================
# SIFT Panorama Stitching
# ============================================================
def stitch_sift(img1, img2):

    # --------------------------------------------------------
    # 1. Gray
    # --------------------------------------------------------
    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # 2. SIFT
    # --------------------------------------------------------
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        gray2,
        None
    )

    print("Image 1 SIFT 特徵點:", len(kp1))
    print("Image 2 SIFT 特徵點:", len(kp2))

    if des1 is None or des2 is None:
        raise ValueError("SIFT 無法找到足夠特徵")


    # --------------------------------------------------------
    # 3. KNN Matching
    # --------------------------------------------------------
    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )


    # --------------------------------------------------------
    # 4. Lowe Ratio Test
    # --------------------------------------------------------
    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)


    print("Good Matches:", len(good_matches))

    if len(good_matches) < 4:
        raise ValueError(
            "有效匹配點少於 4 個，無法計算 Homography"
        )


    # --------------------------------------------------------
    # 5. 取得匹配點座標
    #
    # img1 -> img2
    # --------------------------------------------------------
    src_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)


    # --------------------------------------------------------
    # 6. RANSAC Homography
    # --------------------------------------------------------
    H, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        4.0
    )

    if H is None:
        raise ValueError(
            "Homography 計算失敗"
        )

    inlier_mask = inlier_mask.ravel()

    print(
        "RANSAC Inliers:",
        np.sum(inlier_mask),
        "/",
        len(inlier_mask)
    )

    print("Homography:")
    print(H)


    # --------------------------------------------------------
    # 7. 顯示 RANSAC 後真正可靠的 matching
    # --------------------------------------------------------
    inlier_matches = []

    for i, m in enumerate(good_matches):

        if inlier_mask[i]:
            inlier_matches.append(m)

    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


    # --------------------------------------------------------
    # 8. 取得兩張圖片尺寸
    # --------------------------------------------------------
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]


    # img1 四角
    corners1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)


    # img2 四角
    corners2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)


    # --------------------------------------------------------
    # 9. 把 img1 四個角轉換到 img2 座標系
    # --------------------------------------------------------
    transformed_corners1 = cv2.perspectiveTransform(
        corners1,
        H
    )


    # --------------------------------------------------------
    # 10. 合併兩張圖片所有 corner
    #
    # 用來計算 Panorama 真正需要多大的 canvas
    # --------------------------------------------------------
    all_corners = np.concatenate(
        (
            transformed_corners1,
            corners2
        ),
        axis=0
    )


    x_min = int(
        np.floor(
            np.min(all_corners[:, 0, 0])
        )
    )

    y_min = int(
        np.floor(
            np.min(all_corners[:, 0, 1])
        )
    )

    x_max = int(
        np.ceil(
            np.max(all_corners[:, 0, 0])
        )
    )

    y_max = int(
        np.ceil(
            np.max(all_corners[:, 0, 1])
        )
    )


    print(
        "Panorama bounds:",
        x_min,
        y_min,
        x_max,
        y_max
    )


    # --------------------------------------------------------
    # 11. Translation Matrix
    #
    # 如果 warp 後出現負座標：
    #
    #     x = -300
    #
    # OpenCV canvas 無法表示，
    # 所以整張 panorama 往右/下平移。
    # --------------------------------------------------------
    tx = -x_min
    ty = -y_min

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)


    panorama_width = x_max - x_min
    panorama_height = y_max - y_min

    print(
        "Panorama size:",
        panorama_width,
        "x",
        panorama_height
    )


    # --------------------------------------------------------
    # 12. Warp img1
    #
    # H : img1 -> img2
    # T : img2 coordinate -> panorama coordinate
    #
    # 所以：
    #
    # panorama_H = T @ H
    # --------------------------------------------------------
    panorama_H = T @ H

    warp1 = cv2.warpPerspective(
        img1,
        panorama_H,
        (
            panorama_width,
            panorama_height
        )
    )


    # --------------------------------------------------------
    # 13. img2 也套 Translation
    #
    # 不再直接 img2 貼在 (0,0)
    # --------------------------------------------------------
    warp2 = cv2.warpPerspective(
        img2,
        T,
        (
            panorama_width,
            panorama_height
        )
    )


    # --------------------------------------------------------
    # 14. 建立兩張圖的 Mask
    # --------------------------------------------------------
    source_mask1 = np.ones(
        (h1, w1),
        dtype=np.uint8
    ) * 255

    source_mask2 = np.ones(
        (h2, w2),
        dtype=np.uint8
    ) * 255


    mask1 = cv2.warpPerspective(
        source_mask1,
        panorama_H,
        (
            panorama_width,
            panorama_height
        ),
        flags=cv2.INTER_NEAREST
    )

    mask2 = cv2.warpPerspective(
        source_mask2,
        T,
        (
            panorama_width,
            panorama_height
        ),
        flags=cv2.INTER_NEAREST
    )


    # --------------------------------------------------------
    # 15. Feather blending
    # --------------------------------------------------------
    panorama = feather_blend(
        warp1,
        warp2,
        mask1,
        mask2
    )


    # --------------------------------------------------------
    # 16. Crop 黑邊
    # --------------------------------------------------------
    union_mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    panorama = crop_panorama(
        panorama,
        union_mask
    )


    return panorama, match_img, H


# ============================================================
# MAIN
# ============================================================

img1_path = r"C:\Users\u1021\Downloads\MMIP\week1\data\20260921_182357.jpg"
img2_path = r"C:\Users\u1021\Downloads\MMIP\week1\data\20260921_175900.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))
def adjust_brightness(img, beta):
    temp = img.astype(np.int16) + beta
    temp = np.clip(temp, 0, 255)
    return temp.astype(np.uint8)

img1 = adjust_brightness(img1, -50)
img2 = adjust_brightness(img2, 50)
# def equalize_color(img):
#     # BGR -> YCrCb
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

#     # 拆成亮度 Y、Cr、Cb
#     y, cr, cb = cv2.split(ycrcb)

#     # 只對亮度做 Histogram Equalization
#     y = cv2.equalizeHist(y)

#     # 合併
#     ycrcb = cv2.merge([y, cr, cb])

#     # YCrCb -> BGR
#     result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

#     return result
def clahe_color(img):
    # BGR -> YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 拆開亮度與色彩 channel
    y, cr, cb = cv2.split(ycrcb)

    # 建立 CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # 只對亮度 Y 做 CLAHE
    y_clahe = clahe.apply(y)

    # 合併回去
    ycrcb_clahe = cv2.merge([
        y_clahe,
        cr,
        cb
    ])

    # YCrCb -> BGR
    result = cv2.cvtColor(
        ycrcb_clahe,
        cv2.COLOR_YCrCb2BGR
    )

    return result
img1 = clahe_color(img1)

img2 = clahe_color(img2)

cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
panorama, match_img, H = stitch_sift(
    img1,
    img2
)
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))
match_img = cv2.resize(match_img, (500, 500))
panorama = cv2.resize(panorama, (500, 500))


cv2.imshow("RANSAC Inlier Matches", match_img)
cv2.imshow("Panorama", panorama)

cv2.imwrite(
    "panorama_result.jpg",
    panorama
)

cv2.waitKey(0)
cv2.destroyAllWindows()
