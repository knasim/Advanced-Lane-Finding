# MAIN PROGRAM
# From a Terminal execute:  python main.py from term
# Advanced Lane Finding
# author:  Khurrum Nasim

import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
from function import *
from moviepy.editor import VideoFileClip


# crossing points on the chessboard to detect
nx = 9 # Num of corners in a row
ny = 6 # Num of corners in a column

# images array stores all the calibration images
images = glob.glob("camera_cal/calibration*.jpg")

# store object and image points from all images
objpoints = [] #3D points
imgpoints = [] #2D points

# generate object points
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinate


# loop and calibrate
for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)

# test of  undistortion on an image
img = cv2.imread('camera_cal/calibration11.jpg')
original_image = img
img_size = (img.shape[1], img.shape[0])

# calibrate camera given points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/undistorted.jpg',dst)

# save calibration result
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb"))

# visualize undistortion
f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
axis1.imshow(img)
axis1.set_title('Source Chessboard Image', fontsize=20)
axis2.imshow(dst)
axis2.set_title('Undistorted Chessboard Image', fontsize=20)
####plt.show()

# apply undistort to a sample image
img_sample = cv2.imread("test_images/test2.jpg")

undistorted_img = cv2.undistort(img_sample, mtx, dist, None, mtx)

# visualize undistortion
f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
axis1.imshow(img_sample)
axis1.set_title('Source Highway Image', fontsize=20)
axis2.imshow(undistorted_img)
axis2.set_title('Undistorted Highway Image', fontsize=20)
####plt.show()

image = undistorted_img

# set sobel kernel size
ksize = 3

# apply thresholding functions
gradx = abs_sobel_threshold(image, orient='x', sobel_kernel=ksize, thresh=(150, 255))
grady = abs_sobel_threshold(image, orient='y', sobel_kernel=ksize, thresh=(70, 200))
magnitude_binary = magnitude_threshold(image, sobel_kernel=ksize, mag_thresh=(70, 255))
direction = array_threshold(image, sobel_kernel=ksize, thresh=(0.1, 1.5))

# combine
combined = np.zeros_like(direction)
combined[((gradx == 1) & (grady == 1)) | ((magnitude_binary == 1) & (direction == 1))] = 1


# display sobel function
f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
axis1.imshow(image)
axis1.set_title('Source Highway Image', fontsize=20)
axis2.imshow(grady, cmap='gray')
axis2.set_title('Highway Gradient Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# display image from the magnitude sobel function
f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
axis1.imshow(image)
axis1.set_title('Source Highway Image', fontsize=20)
axis2.imshow(magnitude_binary, cmap='gray')
axis2.set_title('Thresholded Magnitude', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()


# display image from the dir soble function
f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
axis1.imshow(image)
axis1.set_title('Source Highway Image', fontsize=20)
axis2.imshow(direction, cmap='gray')
axis2.set_title('Thresholded Grad. Dir.', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# combine
f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
axis1.imshow(image)
axis1.set_title('Source Highway Image', fontsize=20)
axis2.imshow(combined, cmap='gray')
axis2.set_title('Thresholded Gradient Direction', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()


# display test function threshold
test_binary = vet_threshold(image)

f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
axis1.imshow(image)
axis1.set_title('Source Highway Image', fontsize=20)
axis2.imshow(test_binary, cmap='gray')
axis2.set_title('Thresholded Gradient Direction', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


top_down, perspective_M = warper(combined, nx, ny, mtx, dist)
f, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
axis1.imshow(image)
axis1.set_title('Source Highway Image', fontsize=20)
axis2.imshow(top_down)
axis2.set_title('Undistorted and Warped Result', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()

td_warped = top_down
# compute the lane lines

# a histogram of the bottom half of the image
histogram = np.sum(td_warped[td_warped.shape[0] // 2:, :], axis=0)
out_img = np.dstack((td_warped, td_warped, td_warped)) * 255
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# select number of sliding windows
nwindows = 9
window_height = np.int(td_warped.shape[0] / nwindows)
nonzero = td_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# current pos per window
leftx_current = leftx_base
rightx_current = rightx_base
margin = 100
minpix = 50

# lists to receive left and right lane pixel indices
left_lane_indxs = []
right_lane_indxs = []

for window in range(nwindows):
    win_y_low = td_warped.shape[0] - (window + 1) * window_height
    win_y_high = td_warped.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2)
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2)
    valid_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    valid_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    left_lane_indxs.append(valid_left_inds)
    right_lane_indxs.append(valid_right_inds)

    if len(valid_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[valid_left_inds]))
    if len(valid_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[valid_right_inds]))

left_lane_indxs = np.concatenate(left_lane_indxs)
right_lane_indxs = np.concatenate(right_lane_indxs)

# Extract left and right line pixel positions
left_x_pixel = nonzerox[left_lane_indxs]
left_y_pixel = nonzeroy[left_lane_indxs]
right_x_pixel = nonzerox[right_lane_indxs]
right_y_pixel = nonzeroy[right_lane_indxs]

# apply second order polynomial
left_poly = np.polyfit(np.asarray(left_y_pixel).astype(float), np.asarray(left_x_pixel).astype(float), 2)
right_poly = np.polyfit(np.asarray(right_y_pixel).astype(float), np.asarray(right_x_pixel).astype(float), 2)

"""
lines = lane_lines(td_warped)
left_poly = lines.get('left_poly')
right_poly = lines.get('right_poly')
left_lane_indxs = lines.get('left_lane_indxs')
right_lane_indxs = lines.get('right_lane_indxs')
nonzerox = lines.get('nonzerox')
nonzeroy = lines.get('nonzerox')
out_img = lines.get('out_img')
"""

# x and y values for plotting
ploty = np.linspace(0, td_warped.shape[0] - 1, td_warped.shape[0])
left_fitx = left_poly[0] * ploty ** 2 + left_poly[1] * ploty + left_poly[2]
right_fitx = right_poly[0] * ploty ** 2 + right_poly[1] * ploty +right_poly[2]

out_img[nonzeroy[left_lane_indxs], nonzerox[left_lane_indxs]] = [255, 0, 0]
out_img[nonzeroy[right_lane_indxs], nonzerox[right_lane_indxs]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)


# create image for line drawing
src = np.float32([[490, 482],[810, 482],
                  [1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0],
                  [1250, 720],[40, 720]])
warped = combined
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# reuse  x and y points
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# draw the lane on warped
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
p_transform = cv2.getPerspectiveTransform(dst, src)
newwarp = cv2.warpPerspective(color_warp, p_transform, (image.shape[1], image.shape[0]))
# combine with orignal
result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
#plt.imshow(result)
#####plt.show()

quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
left_x_pixel = left_fitx
right_x_pixel = right_fitx

# apply second order polynomial to pixel positions
left_poly = np.polyfit(np.asarray(ploty).astype(float), np.asarray(left_x_pixel).astype(float), 2)
left_fitx = left_poly[0] * ploty ** 2 + left_poly[1] * ploty + left_poly[2]
right_poly = np.polyfit(np.asarray(ploty).astype(float), np.asarray(right_x_pixel).astype(float), 2)
right_fitx = right_poly[0] * ploty ** 2 + right_poly[1] * ploty + right_poly[2]

# plot fake data
mark_size = 3
plt.plot(left_x_pixel, ploty, 'o', color='red', markersize=mark_size)
plt.plot(right_x_pixel, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
####plt.show()

# y-value radius of curvature
y_eval = np.max(ploty)
left_curve_radius = ((1 + (2 * left_poly[0] * y_eval + left_poly[1]) ** 2) ** 1.5) / np.absolute(2 * left_poly[0])
right_curve_radius = ((1 + (2 * right_poly[0] * y_eval + right_poly[1]) ** 2) ** 1.5) / np.absolute(2 * right_poly[0])
print(left_curve_radius, right_curve_radius)

# define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# apply new polynomials to x,y in world space
left_fit_space = np.polyfit(np.asarray(ploty * ym_per_pix).astype(float), np.asarray(left_x_pixel * xm_per_pix).astype(float), 2)
right_fit_space = np.polyfit(np.asarray(ploty * ym_per_pix).astype(float), np.asarray(right_x_pixel * xm_per_pix).astype(float), 2)
# new radii of curvature
left_curve_radius = ((1 + (2 * left_fit_space[0] * y_eval * ym_per_pix + left_fit_space[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_space[0])
right_curve_radius = ((1 + (2 * right_fit_space[0] * y_eval * ym_per_pix + right_fit_space[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_space[0])
print(left_curve_radius, 'm', right_curve_radius, 'm')


# resulting output video
output = "output.mp4"
clip3 = VideoFileClip("project_video.mp4")#.subclip(20,45)
output_clip = clip3.fl_image(build)
output_clip.write_videofile(output, audio=False)