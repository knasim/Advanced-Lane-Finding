import numpy as np
import cv2
import pickle


nx = 9 # Num of corners per row
ny = 6 # Num of corners per column


def abs_sobel_threshold(img, orient, sobel_kernel, thresh):
    """
    :param img:
    :param orient:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absolute = np.absolute(sobel)
    scaled_sobel = np.uint8(255*absolute/np.max(absolute))
    output = np.zeros_like(scaled_sobel)
    output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return output


def magnitude_threshold(img, sobel_kernel, mag_thresh):
    """
    :param img:
    :param sobel_kernel:
    :param mag_thresh:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    pos_sqrt = np.sqrt(x**2 + y**2)
    scale_factor = np.max(pos_sqrt)/255
    pos_sqrt = (pos_sqrt/scale_factor).astype(np.uint8)
    output = np.zeros_like(pos_sqrt)
    output[(pos_sqrt >= mag_thresh[0]) & (pos_sqrt <= mag_thresh[1])] = 1
    return output


def array_threshold(img, sobel_kernel, thresh):
    """
    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    angle_arr_radians = np.arctan2(np.absolute(y), np.absolute(x))
    output = np.zeros_like(angle_arr_radians)
    output[(angle_arr_radians > thresh[0]) & (angle_arr_radians < thresh[1])] = 1
    return output



def vet_threshold(img):
    """
    :param img:
    :return:
    """
    # Convert image to appropriate channels
    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]
    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]

    # For the L channel
    _l = np.zeros_like(l_channel)
    _l[(l_channel >= 215) & (l_channel <= 255)] = 1

    # For the B channel
    _b = np.zeros_like(b_channel)
    _b[(b_channel >= 145) & (b_channel <= 200)] = 1

    # Combine the two images
    output = np.zeros_like(_b)
    output[(_l == 1) | (_b == 1)] = 1

    return output


def warper(img, nx, ny, mtx, dist):
    """
    receives image, number of x and y points,
    camera matrix and distortion coefficients
    :param img:
    :param nx:
    :param ny:
    :param mtx:
    :param dist:
    :return:
    """
    # get the image shape
    img_size = (img.shape[1], img.shape[0])
    # get detected corners
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                     [1250, 720],[40, 720]])
    # calculate perspective transform
    m = cv2.getPerspectiveTransform(src, dst)
    # warp image
    output = cv2.warpPerspective(img, m, img_size)

    return output, m




def build(file):
    """
    :param file:
    :return:
    """

    from main import combined
    global prev_right_y
    global prev_right_x

    with open("camera_cal/wide_dist_pickle.p", mode='rb') as f:
        camera_calib = pickle.load(f)
    mtx = camera_calib["mtx"]
    dist = camera_calib["dist"]

    prev_right_y = None
    prev_right_x = None

    # select sobel kernel size
    test_binary = vet_threshold(file)
    top_down, perspective_m = warper(test_binary, nx, ny, mtx, dist)
    binary_warped = top_down

    # histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # produce output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # determine peak of the left and right halves of the histogram
    # this will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # number of sliding windows
    nwindows = 9
    # height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # x and y positions of all nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # select the width of the windows +/- margin
    margin = 100
    # select minimum number of pixels
    minpix = 50
    # create empty lists
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # extract left and right pixels
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]
    if right_y.size == 0:
        right_y = prev_right_y
        right_x = prev_right_x
    # fit a second order polynomial
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    prev_right_y = right_y
    prev_right_x = right_x

    # find the pixels
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y +
                                   left_fit[2] - margin)) & (nonzero_x < (left_fit[0] * (nonzero_y ** 2) +
                                                                         left_fit[1] * nonzero_y + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y +
                                    right_fit[2] - margin)) & (nonzero_x < (right_fit[0] * (nonzero_y ** 2) +
                                                                           right_fit[1] * nonzero_y + right_fit[
                                                                               2] + margin)))

    # extract left and right line pixels
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]
    # apply second order polynomial
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    # generate x and y values for plotting
    plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    # create image to draw the lines on
    src = np.float32([[490, 482], [810, 482],
                      [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])
    warped = combined
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    print(color_warp.shape)
    # x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # draw the lane (using warped image)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    Minv = cv2.getPerspectiveTransform(dst, src)
    # warp the blank back to original image
    new_warp = cv2.warpPerspective(color_warp, Minv, (file.shape[1], file.shape[0]))
    # combine results
    result_output = cv2.addWeighted(file, 1, new_warp, 0.3, 0)

    # generate random x position within +/-50 pix
    left_x = left_fitx
    right_x = right_fitx

    # apply  second order polynomial to pixels
    y_eval = np.max(plot_y)

    ym_per_pix = 30 / 720  # meters per pixel y axis
    xm_per_pix = 3.7 / 700  # meters per pixel x axis

    # apply polynomial to x,y
    left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_x * xm_per_pix, 2)
    # calc. new radii
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # print curvature
    curvature = (left_curverad + right_curverad) / 2
    cv2.putText(result_output, 'Curvature Radius {}(m)'.format(curvature), (120, 140),fontFace=16, fontScale=2, color=(255, 255, 255))

    center = (right_x[0] + left_x[0]) / 2
    dist_center = abs((640 - center) * 3.7 / 700)
    if center > 640:
        cv2.putText(result_output, 'Vehicle is {:.2f}m left of center'.format(dist_center), (100, 80),fontFace=16, fontScale=2, color=(255, 255, 255))
    else:
        cv2.putText(result_output, 'Vehicle is {:.2f}m right of center'.format(dist_center), (100, 80),fontFace=16, fontScale=2, color=(255, 255, 255))

    return result_output