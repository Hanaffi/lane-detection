import glob
from math import floor

import numpy as np
import cv2

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
