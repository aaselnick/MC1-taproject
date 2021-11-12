import cv2 as cv
import numpy as np
import glob
import time

#termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2]=np.mgrid[0:7,0:6].T.reshape(-1,2)

#arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] #2d points in image plane.

time.sleep(2)

cap = cv.VideoCapture(0)
for i in range(1,10):
    ret, img = cap.read()
    cv.imwrite("img" + str(i) + '.jpg', img)
    print('took' + str(i) + " pictures")
    time.sleep(7)

cap.release()

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    retval, corners = cv.findChessboardCorners(gray, (7,6), None)

    if retval == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img,(7,6),corners2,ret)
        cv.imshow('img',img)

retval2, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread('img9.jpg')
h,w=img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv.undistort(img,mtx,dist,None,newcameramtx)

x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)


k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()

