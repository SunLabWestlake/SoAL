"""
Calibration and anti-distortion
Author: Jing Ning @ SunLab
"""

import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

def calib(path):
    corner_s = (9, 7)
    prefix = path[:path.rfind(".")]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((corner_s[0] * corner_s[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_s[0], 0:corner_s[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    img = cv2.imread(path)
    gray = img[:, :, 0]
    gray = (gray>90).astype(np.uint8)*255
    plt.imshow(gray, cmap="Greys_r")
    plt.show()
    # return
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, corner_s, None)
    # If found, add object points, image points (after refining them)
    if not ret:
        print("no corner found")
        return
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    cv2.drawChessboardCorners(gray, corner_s, corners2, ret)
    plt.imshow(gray, cmap="Greys_r")
    plt.show()
    # return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # NOTE: save calib info
    store_info = [mtx, dist, newcameramtx, (w, h)]
    pickle.dump(store_info, open(prefix + "_info.pickle", "wb"))

    xs = corners2[:,:,0]
    ys = corners2[:,:,1]
    u_points = cv2.undistortPoints(corners2, mtx, dist, P=newcameramtx)  # NOTE: use this function to compute real pos
    remap_xs = u_points[:,:,0]
    remap_ys = u_points[:,:,1]
    # undistort
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    # NOTE: dst[newx, newy] = gray[mapx[newx, newy], mapy[newx, newy]]
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    cv2.imwrite(prefix + "_result.png", dst)

    plt.imshow(gray, cmap="Greys_r")
    plt.scatter(xs, ys, c="r", s=15, marker="x")
    plt.scatter(remap_xs, remap_ys, c="g", s=15, marker="+")
    plt.show()

def test_undistort(video, frame, point, calib_info_pickle):
    calib_info = pickle.load(open(calib_info_pickle, "rb"))
    mtx, dist, newcameramtx, sz = calib_info

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    img_gray = img[:, :, 1]
    u_points = cv2.undistortPoints(np.reshape(point, (1, 1, 2)), mtx, dist, P=newcameramtx)
    plt.imshow(img_gray, cmap="Greys_r")
    plt.scatter([point[0]], [point[1]], c="r", s=15, marker="x")
    plt.scatter([u_points[0][0][0]], [u_points[0][0][1]], c="g", s=15, marker="+")
    plt.show()

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, sz, 5)
    dst = cv2.remap(img_gray, mapx, mapy, cv2.INTER_LINEAR)
    plt.imshow(dst, cmap="Greys_r")
    plt.show()

if __name__ == '__main__':
    calib("calib_1.jpg")
    #test_undistort(sys.argv[1], 0, (32.17+103, 74.61+549), "calib_1.pickle")
