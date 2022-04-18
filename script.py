import sys
import glob
from math import floor

import numpy as np
import cv2

from moviepy.editor import VideoFileClip

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
nx = 9
ny = 6
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob("camera_cal/calibration*.jpg")

# dummy vals
img, gray = None, None

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        write_name = "camera_cal/corners_marked" + str(idx) + ".jpg"
        cv2.imwrite(write_name, img)

# use our set of image points and object points to approximate camera distortion matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)


# input an image
# output undistorted image
def distortion_correction(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# input a color or gray scale image of the road (1280*720)
# output the bird_view the road, for finding lane line
# if reverse = True, it converts bird view image back to original perspective
def warper(img, reverse=False):
    xsize = img.shape[1]
    ysize = img.shape[0]

    # set up box boundary
    xmid = xsize / 2  # middle point
    upper_margin = 85  # upper width
    lower_margin = 490  # lower width
    upper_bound = 460  # upper value of y
    lower_bound = 670  # bottom value of y
    dst_margin = 450  # bird view width

    # source points
    p1_src = [xmid - lower_margin, lower_bound]
    p2_src = [xmid - upper_margin, upper_bound]
    p3_src = [xmid + upper_margin, upper_bound]
    p4_src = [xmid + lower_margin, lower_bound]
    src = np.array([p1_src, p2_src, p3_src, p4_src], dtype=np.float32)

    # distination points
    p1_dst = [xmid - dst_margin, ysize]
    p2_dst = [xmid - dst_margin, 0]
    p3_dst = [xmid + dst_margin, 0]
    p4_dst = [xmid + dst_margin, ysize]
    dst = np.array([p1_dst, p2_dst, p3_dst, p4_dst], dtype=np.float32)

    if not reverse:
        # if we need to change to bird view
        # given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        # else switch src and dst, change back from bird view
        M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    return warped


# applying sobel operators and HSV color transforms.
def combined_binary(img):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(
        sobelx
    )  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


# defining our line class
class Line:
    def __init__(self):
        # have we seen this lane before?
        self.detected = False

        # previous n lines
        self.recent_left_fit = []
        self.recent_right_fit = []

        # coefficients of the most recent fit
        self.current_left_fit = [np.array([False])]
        self.current_right_fit = [np.array([False])]

    # get the best coefficients, average the last 3
    def average_fit(self):
        cur_left_fit = np.average(self.recent_left_fit[-3:], axis=0)
        cur_right_fit = np.average(self.recent_right_fit[-3:], axis=0)
        return cur_left_fit, cur_right_fit

    # calculate the offset based on the lines
    # Attempt to confirm that the center of the lanes is in the center of the image.
    def find_offset(self):
        left_fit, right_fit = self.average_fit()
        base = 719  # given in lesson
        middle = 1279 / 2  # middle of the image
        xm_per_pix = 3.7 / 650  # meters per pixel in x dimension, given in the lesson
        left_base = left_fit[0] * base**2 + left_fit[1] * base + left_fit[2]
        right_base = right_fit[0] * base**2 + right_fit[1] * base + right_fit[2]
        offset = ((right_base - left_base) - middle) * xm_per_pix
        return offset

    # return curvature based on method given in lesson.
    def find_curvature(self):
        left_fit, right_fit = self.average_fit()

        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)  # cover same range as the image

        # arbitrary quadratic coefficient
        l_quadratic_coeff = left_fit[0]
        r_quadratic_coeff = right_fit[0]

        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array(
            [
                200 + (y**2) * l_quadratic_coeff + np.random.randint(-50, high=51)
                for y in ploty
            ]
        )
        rightx = np.array(
            [
                900 + (y**2) * r_quadratic_coeff + np.random.randint(-50, high=51)
                for y in ploty
            ]
        )

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane line

        # fit to the left line
        left_fit = np.polyfit(ploty, leftx, 2)

        # fit to the right line
        right_fit = np.polyfit(ploty, rightx, 2)

        # Define y value where we want radius of curvature
        # in this case we'll use the max y value for the bottom of the image.
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 650  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature (in meters)
        left_curverad = (
            (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2)
            ** 1.5
        ) / np.absolute(2 * left_fit_cr[0])
        right_curverad = (
            (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2)
            ** 1.5
        ) / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def find_lane(self, warped):
        if not self.detected:
            # do our bad search if we've never seen the lines before
            detect = self.blind_search(warped)
        else:
            # do our better search based on what we already know.
            detect = self.margin_search(warped)

        left_fit = self.current_left_fit
        right_fit = self.current_right_fit

        # sanity check pipeline
        # check base distance
        base = 720
        left_base = left_fit[0] * base**2 + left_fit[1] * base + left_fit[2]
        right_base = right_fit[0] * base**2 + right_fit[1] * base + right_fit[2]

        if right_base < left_base or np.absolute((right_base - left_base) - 650) >= 200:
            detect = False

        # check derivative difference
        if (
            np.absolute(
                (2 * left_fit[0] + left_fit[1]) - (2 * right_fit[0] + right_fit[1])
            )
            >= 0.8
        ):
            detect = False

        # force to append the first few detection
        if len(self.recent_left_fit) < 3 or len(self.recent_right_fit) < 3:
            detect = False
            self.recent_left_fit.append(left_fit)
            self.recent_right_fit.append(right_fit)

        # if it is a real dection, we add to recent detection
        if detect:
            self.detected = True
            self.recent_left_fit.append(left_fit)
            self.recent_right_fit.append(right_fit)
        else:
            self.detected = False

    # use moving histograms to search for new lines
    def blind_search(self, warped):
        # Assuming you have created a warped binary image called "binary_warped"
        binary_warped = np.sum(warped, axis=2)  # collapse 3 channel into 1
        binary_y = warped[:, :, 0]
        binary_w = warped[:, :, 1]

        # Take a histogram of the bottom half of the image
        histogram_l = np.sum(binary_y[floor(binary_y.shape[0] / 2):, :], axis=0)
        histogram_r = np.sum(binary_w[floor(binary_w.shape[0] / 2):, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(floor(histogram_l.shape[0] / 2))
        leftx_base = np.argmax(histogram_l[:midpoint])
        rightx_base = np.argmax(histogram_r[midpoint:]) + midpoint

        nwindows = 9
        margin = 100
        minpix = 50
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero_l = binary_y.nonzero()
        nonzeroy_l = np.array(nonzero_l[0])
        nonzerox_l = np.array(nonzero_l[1])

        nonzero_r = binary_w.nonzero()
        nonzeroy_r = np.array(nonzero_r[0])
        nonzerox_r = np.array(nonzero_r[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        leftx = []
        lefty = []
        rightx = []
        righty = []

        # points of current fit
        cur_left_fit, cur_right_fit = self.average_fit()

        # Step through the windows one by one
        for window in range(nwindows):
            # find windows bounds in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(
                warped,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                warped,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (0, 255, 0),
                2,
            )

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = (
                (nonzeroy_l >= win_y_low)
                & (nonzeroy_l < win_y_high)
                & (nonzerox_l >= win_xleft_low)
                & (nonzerox_l < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy_r >= win_y_low)
                & (nonzeroy_r < win_y_high)
                & (nonzerox_r >= win_xright_low)
                & (nonzerox_r < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox_l[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox_r[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox_l[left_lane_inds]
        lefty = nonzeroy_l[left_lane_inds]

        rightx = nonzerox_r[right_lane_inds]
        righty = nonzeroy_r[right_lane_inds]

        detect = True
        if len(leftx) > 1000 or len(self.recent_left_fit) < 1:
            self.current_left_fit = np.polyfit(lefty, leftx, 2)

        else:  # not enough point to suggest a good fit, use last fit
            print("left blind search fail")
            self.current_left_fit = cur_left_fit
            detect = False

        if len(rightx) > 1000 or len(self.recent_right_fit) < 1:
            self.current_right_fit = np.polyfit(righty, rightx, 2)
        else:
            # not enough point to suggest a good fit, use last fit
            print("right blind search fail")
            self.current_right_fit = cur_right_fit
            detect = False
        return detect

    # use best fit coefficients to restrict search area within margin
    def margin_search(self, warped):

        # Assuming you have created a warped binary image called "binary_warped"
        binary_warped = np.sum(warped, axis=2)  # collapse 3 channel into 1
        binary_y = warped[:, :, 0]
        binary_w = warped[:, :, 1]

        nwindows = 9
        margin = 100
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Set minimum number of pixels found to recenter window
        minpix = 50
        maxpix = 6000

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero_l = binary_y.nonzero()
        nonzeroy_l = np.array(nonzero_l[0])
        nonzerox_l = np.array(nonzero_l[1])

        nonzero_r = binary_w.nonzero()
        nonzeroy_r = np.array(nonzero_r[0])
        nonzerox_r = np.array(nonzero_r[1])

        leftx = []
        lefty = []
        rightx = []
        righty = []

        # points of current fit
        cur_left_fit, cur_right_fit = self.average_fit()
        yvals = np.linspace(
            0, 719, num=720, dtype=np.int32
        )  # to cover same y-range as image
        l_xvals = []
        r_xvals = []
        for y in yvals:
            l_xvals.append(
                cur_left_fit[0] * y**2
                + cur_left_fit[1] * y
                + cur_left_fit[2]
                + np.random.randint(-50, high=51)
            )
            r_xvals.append(
                cur_right_fit[0] * y**2
                + cur_right_fit[1] * y
                + cur_right_fit[2]
                + np.random.randint(-50, high=51)
            )
        l_xvals = np.array(l_xvals, dtype=np.int32)
        r_xvals = np.array(r_xvals, dtype=np.int32)
        # Step through the windows one by one
        l_base_missing = 0
        r_base_missing = 0

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            leftx_current = np.int(
                cur_left_fit[0] * win_y_high**2
                + cur_left_fit[1] * win_y_high
                + cur_left_fit[2]
            )
            rightx_current = np.int(
                cur_right_fit[0] * win_y_high**2
                + cur_right_fit[1] * win_y_high
                + cur_right_fit[2]
            )

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(
                warped,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (0, 255, 0),
                4,
            )
            cv2.rectangle(
                warped,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (0, 255, 0),
                4,
            )
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = (
                (nonzeroy_l >= win_y_low)
                & (nonzeroy_l < win_y_high)
                & (nonzerox_l >= win_xleft_low)
                & (nonzerox_l < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy_r >= win_y_low)
                & (nonzeroy_r < win_y_high)
                & (nonzerox_r >= win_xright_low)
                & (nonzerox_r < win_xright_high)
            ).nonzero()[0]

            # check left side
            # if the number of points within a reasonable range, it suggest a good detection
            # too less or too much points suggest noise
            if (len(good_left_inds) > minpix) and (len(good_left_inds) < maxpix):
                leftx.append(nonzerox_l[good_left_inds])
                lefty.append(nonzeroy_l[good_left_inds])
            else:
                # use last fit to generate fake points
                # means when fail to search in this window,
                # we guess it should have lane line similar to last detection within the same window position
                good_left_inds = (
                    (yvals >= win_y_low)
                    & (yvals < win_y_high)
                    & (l_xvals >= win_xleft_low)
                    & (l_xvals < win_xleft_high)
                ).nonzero()[0]
                if window <= 4:
                    l_base_missing = l_base_missing + 1
                    leftx.append(l_xvals[good_left_inds])
                    lefty.append(yvals[good_left_inds])
                    # if use fake data, we plot it
                    for p in good_left_inds:
                        cv2.circle(warped, (l_xvals[p], yvals[p]), 3, (0, 255, 255))
                elif l_base_missing >= 1:
                    leftx.append(l_xvals[good_left_inds])
                    lefty.append(yvals[good_left_inds])
                    # if use fake data, we plot it
                    for p in good_left_inds:
                        cv2.circle(warped, (l_xvals[p], yvals[p]), 3, (0, 255, 255))

            # check right side
            # same idea as checking left side
            if (len(good_right_inds) > minpix) and (len(good_right_inds) <= maxpix):
                rightx.append(nonzerox_r[good_right_inds])
                righty.append(nonzeroy_r[good_right_inds])
            else:
                good_right_inds = (
                    (yvals >= win_y_low)
                    & (yvals < win_y_high)
                    & (r_xvals >= win_xright_low)
                    & (r_xvals < win_xright_high)
                ).nonzero()[0]
                if window <= 4:
                    r_base_missing = r_base_missing + 1
                    rightx.append(r_xvals[good_right_inds])
                    righty.append(yvals[good_right_inds])
                    for p in good_right_inds:
                        cv2.circle(warped, (r_xvals[p], yvals[p]), 5, (0, 0, 255))
                elif r_base_missing >= 1:
                    rightx.append(r_xvals[good_right_inds])
                    righty.append(yvals[good_right_inds])
                    for p in good_right_inds:
                        cv2.circle(warped, (r_xvals[p], yvals[p]), 5, (0, 0, 255))

        leftx = np.concatenate(leftx)
        lefty = np.concatenate(lefty)
        self.current_left_fit = np.polyfit(lefty, leftx, 2)

        rightx = np.concatenate(rightx)
        righty = np.concatenate(righty)
        self.current_right_fit = np.polyfit(righty, rightx, 2)

        return True


def draw_lane(undist, warped, stack, left_fit, right_fit):
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    warped_mask = warped.copy()
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(stack[:, :, 0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
    cv2.polylines(stack, np.int_([pts]), True, (255, 255, 255), 5)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warper(color_warp, reverse=True)

    # Combine the result with the original image
    warped_mask = cv2.addWeighted(warped_mask, 1, color_warp, 0.3, 0)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result, warped_mask


# find white lane lines
def find_white_lanes(img):
    img = np.copy(img)
    # convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 3 different threshold for different lighting condition

    # high light , color = B, used in normal light condition
    lower_1 = np.array([0, 0, 200])
    upper_1 = np.array([255, 25, 255])

    # high light , color = G, not used, saved for future experiment
    lower_2 = np.array([0, 0, 200])
    upper_2 = np.array([255, 25, 255])

    # low h low s low v , color = R, used in low light condition
    lower_3 = np.array([0, 0, 170])
    upper_3 = np.array([255, 20, 190])

    white_1 = cv2.inRange(hsv, lower_1, upper_1)
    white_2 = cv2.inRange(hsv, lower_2, upper_2)
    white_3 = cv2.inRange(hsv, lower_3, upper_3)

    if len(white_1.nonzero()[0]) > 4000:
        white_3 = np.zeros_like(white_1)

    if len(white_2.nonzero()[0]) > 40000:  # too much false detection
        white_2 = np.zeros_like(white_1)

    if len(white_3.nonzero()[0]) > 40000:  # too much false detection
        white_3 = np.zeros_like(white_1)

    return white_1, white_2, white_3


# find yellow lane lines
def find_yellow_lanes(img):
    img = np.copy(img)
    # convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)

    # color = B, high s high v, used in normal light condition
    lower_1 = np.array([0, 100, 100])
    upper_1 = np.array([50, 255, 255])

    # color = G, low s low v, used in low light condition
    lower_2 = np.array([10, 35, 100])
    upper_2 = np.array([40, 80, 180])

    # color = R, low s high v, used in extreme high light condition
    lower_3 = np.array([15, 30, 150])
    upper_3 = np.array([45, 80, 255])

    yellow_1 = cv2.inRange(hsv, lower_1, upper_1)
    yellow_2 = cv2.inRange(hsv, lower_2, upper_2)
    yellow_3 = cv2.inRange(hsv, lower_3, upper_3)

    if len(yellow_1.nonzero()[0]) > 30000:
        yellow_2 = np.zeros_like(yellow_1)
        yellow_3 = np.zeros_like(yellow_1)

    if len(yellow_2.nonzero()[0]) > 30000:
        yellow_2 = np.zeros_like(yellow_1)

    if len(yellow_3.nonzero()[0]) > 30000:
        yellow_3 = np.zeros_like(yellow_1)

    return yellow_1, yellow_2, yellow_3


# full pipeline function
def final_pipeline(img):
    global tracker
    global frame_id
    frame_id = frame_id + 1

    # image is input in RGB format from moviepy
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # undistort the image
    undist = distortion_correction(img, mtx, dist)

    # warp to bird view
    warped = warper(undist, reverse=False)

    # find yellow and white lane line
    # and map to binary image
    yellow_1, yellow_2, yellow_3 = find_yellow_lanes(warped)
    white_1, white_2, white_3 = find_white_lanes(warped)

    w = np.zeros_like(white_1)
    w[(white_1 > 0) | (white_2 > 0) | (white_3 > 0)] = 255

    y = np.zeros_like(yellow_1)
    y[(yellow_1 > 0) | (yellow_2 > 0) | (yellow_3 > 0)] = 255

    # stack yellow and white lane line into one image
    stack = np.zeros_like(img)
    stack[:, :, 0] = y
    stack[:, :, 1] = w

    # find lane lines
    tracker.find_lane(stack)

    # get best fitting coefficents
    left_fit, right_fit = tracker.average_fit()

    # draw the road mask
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    result, warped_mask = draw_lane(undist, warped, stack, left_fit, right_fit)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # get statistics
    left_curverad, right_curverad = tracker.find_curvature()
    offset = tracker.find_offset()

    # round our numbers for the display purposes
    left_curverad = round(left_curverad, 2)
    right_curverad = round(right_curverad, 2)
    offset = round(float(offset), 2)

    # text font and color in video
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)  # white for overwriting video

    text_panel = np.zeros((240, 640, 3), dtype=np.uint8)

    # frame id
    cv2.putText(
        text_panel,
        "Frame id:     " + str(frame_id),
        (30, 40),
        text_font,
        1,
        text_color,
        2,
    )

    # left curve radius
    cv2.putText(
        text_panel,
        "Left Curve Radius: " + str(left_curverad) + "m",
        (30, 80),
        text_font,
        1,
        text_color,
        2,
    )

    # right curve radius
    cv2.putText(
        text_panel,
        "Right Curve Radius: " + str(right_curverad) + "m",
        (30, 120),
        text_font,
        1,
        text_color,
        2,
    )

    # middle panel radius
    cv2.putText(
        text_panel,
        "Center Offset:" + str(offset) + "m",
        (30, 160),
        text_font,
        1,
        text_color,
        2,
    )

    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    warped = cv2.resize(warped, (320, 240), interpolation=cv2.INTER_AREA)
    warped_plot = cv2.resize(stack, (320, 240), interpolation=cv2.INTER_AREA)
    warped_mask = cv2.resize(warped_mask, (640, 480), interpolation=cv2.INTER_AREA)

    # render original video with lane segment drawn in
    diagScreen[0:720, 0:1280] = result
    diagScreen[0:480, 1280:1920] = warped_mask
    diagScreen[480:720, 1600:1920] = warped_plot
    diagScreen[480:720, 1280:1600] = warped
    diagScreen[720:960, 1280:1920] = text_panel

    return diagScreen


if __name__ == "__main__":
    # frame id
    tracker = Line()
    frame_id = 0

    # read in the project video
    input_video_filename = sys.argv[1]
    output_video_filename = sys.argv[2]

    clip = VideoFileClip(input_video_filename)
    clip = clip.fl_image(final_pipeline)
    clip.write_videofile(output_video_filename, audio=False)
