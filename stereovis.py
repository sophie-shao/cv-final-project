import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature
from skimage.measure import regionprops
import numpy as np
import cv2
import random
from skimage import io, img_as_float32
from skimage.color import rgb2gray
from skimage.transform import rescale

def test_stereo_matching(image1_file, image2_file):
    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))
    image1 = rgb2gray(image1)
    # Our own rgb2gray coefficients which match Rec.ITU-R BT.601-7 (NTSC) luminance conversion - only mino performance improvements and could be confusing to students
    # image1 = image1[:,:,0] * 0.2989 + image1[:,:,1] * 0.5870 + image1[:,:,2] * 0.1140
    image2 = rgb2gray(image2)
    # image2 = image2[:,:,0] * 0.2989 + image2[:,:,1] * 0.5870 + image2[:,:,2] * 0.1140

    # make images smaller to speed up the algorithm. This parameter
    # gets passed into the evaluation code, so don't resize the images
    # except for changing this parameter - We will evaluate your code using
    # scale_factor = 0.5, so be aware of this
    scale_factor = 0.5

    # Bilinear rescaling
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    # width and height of each local feature, in pixels
    feature_width = 16

    # (2) Find distinctive points in each image. See Szeliski 4.1.1
    # !!! You will need to implement get_feature_points. !!!

    print("Getting feature points...")

    (x1, y1) = get_feature_points(image1,feature_width)
    (x2, y2) = get_feature_points(image2,feature_width)


    # Viewing your feature points on your images.
    # !!! You will need to implement plot_feature_points. !!!
    print("Number of feature points found (image 1):", len(x1))
    plot_feature_points(image1, x1, y1)
    print("Number of feature points found (image 2):", len(x2))
    plot_feature_points(image2, x2, y2)

    print("Done!")

    print("Getting features...")

    mode = "sift"
    image1_features = get_feature_descriptors(image1, x1, y1, feature_width, mode)
    image2_features = get_feature_descriptors(image2, x2, y2, feature_width, mode)

    print("Done!")

    # 4) Match features. Szeliski 4.1.3
    # !!! You will need to implement match_features !!!

    print("Matching features...")

    matches = match_features(image1_features, image2_features)

    print("Done!") # might not need this function after all! see main.py in hw 2, which allows for custom image

#----------------------------------------------------------------------------------------#

# for homework 3 functions
inlier_counts = []
inlier_residuals = []

def plot_feature_points(image, xs, ys):
    '''
    Plot feature points for the input image. 
    
    Show the feature points (x, y) over the image. Be sure to add the plots you make to your writeup!

    Useful functions: Some helpful (but not necessarily required) functions may include:
        - plt.imshow
        - plt.scatter
        - plt.show
        - plt.savefig
    
    :params:
    :image: a grayscale or color image (depending on your implementation)
    :xs: np.array of x coordinates of feature points
    :ys: np.array of y coordinates of feature points
    '''

    # TODO[COMPLETED*]: Your implementation here!
    plt.imshow(image, cmap='gray')
    plt.scatter(xs, ys, facecolors='none', edgecolors='b')
    # plt.savefig("plot_student_writeup.png")
    plt.show()

def get_feature_points(image, window_width):
    '''
    Implement the Harris corner detector to return feature points for a given image.

    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.

    If you're finding spurious (false/fake) feature point detections near the boundaries,
    it is safe to suppress the gradients / corners near the edges of the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_feature_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :window_width: the width and height of each local window in pixels

    :returns:
    :xs: an np.array of the x coordinates (column indices) of the feature points in the image
    :ys: an np.array of the y coordinates (row indices) of the feature points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np.array indicating the confidence (strength) of each feature point
    :scale: an np.array indicating the scale of each feature point
    :orientation: an np.array indicating the orientation of each feature point

    '''

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    gradient = np.gradient(image)
    # STEP 2: Apply Gaussian filter with appropriate sigma. NOTE 10/16: Settled on sigma=1.5
    sigma_value = 3 # higher value might be good too
    g_Ix_2 = filters.gaussian(gradient[0] * gradient[0], sigma=sigma_value)
    g_Iy_2 = filters.gaussian(gradient[1] * gradient[1], sigma=sigma_value)
    g_Ix_Iy = filters.gaussian(gradient[0] * gradient[1], sigma=sigma_value)
    
    # STEP 3: Calculate Harris cornerness score for all pixels.
    alpha = 0.05 # can be anywhere between 0.04 and 0.06, according to the lecture slides
    C = (g_Ix_2 * g_Iy_2) - (g_Ix_Iy**2) - alpha * ((g_Ix_2 + g_Iy_2)**2)
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    peak_array = feature.peak_local_max(C, min_distance=3, threshold_rel=0.04, exclude_border=int(window_width/2), num_peaks=1000)
    xs = peak_array[:,1] # is it like this or the other way around?
    ys = peak_array[:,0]

    # TODO [COMPLETED*]: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    # xs = np.random.randint(0, image.shape[1], size=100)
    # ys = np.random.randint(0, image.shape[0], size=100)

    return xs, ys


def get_feature_descriptors(image, xs, ys, window_width, mode):
    '''
    Computes features for a given set of feature points.

    To start with, use image patches as your local feature descriptor. You will 
    then need to implement the more effective SIFT-like feature descriptor. Use 
    the `mode` argument to toggle between the two.
    (Original SIFT publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) A 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) Each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4 x 4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    This is a design task, so many options might help but are not essential.
    - To perform interpolation such that each gradient
    measurement contributes to multiple orientation bins in multiple cells
    A single gradient measurement creates a weighted contribution to the 4 
    nearest cells and the 2 nearest orientation bins within each cell, for 
    8 total contributions.

    - To compute the gradient orientation at each pixel, we could use oriented 
    kernels (e.g. a kernel that responds to edges with a specific orientation). 
    All of your SIFT-like features could be constructed quickly in this way.

    - You could normalize -> threshold -> normalize again as detailed in the 
    SIFT paper. This might help for specular or outlier brightnesses.

    - You could raise each element of the final feature vector to some power 
    that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.filters (library)

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :xs: np.array of x coordinates (column indices) of feature points
    :ys: np.array of y coordinates (row indices) of feature points
    :window_width: in pixels, is the local window width. You can assume
                    that window_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like window will have an integer width and height).
    :mode: a string, either "patch" or "sift". Switches between image patch descriptors
           and SIFT descriptors

    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np.array of computed features. features[i] is the descriptor for 
               point (x[i], y[i]), so the shape of features should be 
               (len(x), feature dimensionality). For standard SIFT, `feature
               dimensionality` is typically 128. `num points` may be less than len(x) if
               some points are rejected, e.g., if out of bounds.
    '''
    # initialize features to the shape we want; fill with correct values later

    if mode == "patch":
        features = np.zeros((len(xs), 256)) 
        # IMAGE PATCH STEPS
        # STEP 1: For each feature point, cut out a window_width x window_width patch 
        #         of the image around that point (as you will in SIFT)
        for i in range(len(xs)):
            adj = int(window_width/2)
            image_patch = image[ys[i]-adj:ys[i]+adj, xs[i]-adj:xs[i]+adj]
        # STEP 2: Flatten this image patch into a 1-dimensional vector (hint: np.flatten())
            flattened_patch = image_patch.flatten()
            features[i,:] = flattened_patch / np.linalg.norm(flattened_patch) # normalize with norm
            
    elif mode == "sift":
        features = np.zeros((len(xs), 128)) 
    # SIFT STEPS
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
        gradient = np.gradient(image) 
    # STEP 2: Decompose the gradient vectors to magnitude and orientation (angle).
        magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        orientation = np.arctan2(gradient[1], gradient[0]) # assuming this does element-wise division
        orientation[orientation < 0] += 2 * np.pi # make all values between 0 and 2pi
        orientation_bins = np.floor(orientation * 4 / np.pi) # all values between 0 and 7 (for histogram bins)
    # STEP 3: For each feature point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the orientation (angle) of the gradient vectors. 
        for i in range(len(xs)): 
            adj = int(window_width/2)
            mag_patch = magnitude[ys[i]-adj:ys[i]+adj, xs[i]-adj:xs[i]+adj] # Computing patch similarly
            ori_patch = orientation_bins[ys[i]-adj:ys[i]+adj, xs[i]-adj:xs[i]+adj] # orientations as well!

            # Start by initializing histograms that will combine to become our feature
            num_cells_wide = int(mag_patch.shape[1]/4) # should = 4 if window_width is 16
            num_cells_tall = int(mag_patch.shape[0]/4) # should = 4 if window_width is 16
            histograms = np.zeros((num_cells_wide, num_cells_tall,8)) # a number of 8-bin histograms

            # Now, for each pixel in the patch, add its magnitude to the appropriate orientation bin
            for j in range(mag_patch.shape[1]):
                for k in range(mag_patch.shape[0]):
                    histograms[int(j//4), int(k//4), int(ori_patch[j,k])] += mag_patch[j,k]

    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
            flattened_feature = histograms.flatten() 
            features[i,:] = flattened_feature / np.linalg.norm(flattened_feature)
    # STEP 5: Don't forget to normalize your feature.

    # TODO [COMPLETED*]: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    else:
        print('Mode not recognized; please try again!')

    return features


def match_features(im1_features, im2_features):
    '''
    Matches feature descriptors of one image with their nearest neighbor in the other.

    Implements the Nearest Neighbor Distance Ratio (NNDR) Test to help threshold
    and remove false matches.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test".

    For extra credit you can implement spatial verification of matches.

    Remember that the NNDR will return a number close to 1 for feature 
    points with similar distances. Think about how you might want to threshold
    this ratio (hint: see lecture slides for NNDR)

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - np.argsort()

    :params:
    :im1_features: an np.array of features returned from get_feature_descriptors() for feature points in image1
    :im2_features: an np.array of features returned from get_feature_descriptors() for feature points in image2

    :returns:
    :matches: an np.array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    '''

    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    
    # OLD CODE: using for loop, which is costly 
    # distance_matrix = np.zeros((im1_features.shape[0], im2_features.shape[0]))
    # for i in range(im1_features.shape[0]):
    #     for j in range(im2_features.shape[0]): # iterating over every possible combo of features
    #         # vector l2 norm between each pair of features
    #         distance_matrix[i,j] = np.linalg.norm(im2_features[j,:] - im1_features[i,:]) 

    A = np.sum(im1_features **2, axis=1)[:, np.newaxis] + np.sum(im2_features**2, axis=1)[np.newaxis, :]
    B = 2 * np.dot(im1_features, np.transpose(im2_features))
    distance_matrix = np.sqrt(A - B)
    # STEP 2: Sort and find closest features for each feature
    argsorted = np.argsort(distance_matrix) # sort across columns
    # Now, the leftmost column in argsorted tells us the column index of the closest neighbor IN im2
    # FOR each feature in im1.

    # STEP 3: Compute NNDR for each match
    matches = np.zeros((im1_features.shape[0],2))
    match_counter = 0 # match_counter tracks the number of matches and should equal the number of matches at the end
    for i in range(argsorted.shape[0]): # i is the feature number in im1_features
        # closest match to ith point in im1 over next closest
        ratio = distance_matrix[i, argsorted[i,0]] / distance_matrix[i, argsorted[i,1]] 
    # STEP 4: Remove matches whose ratios do not meet a certain threshold 
        if ratio <= 0.85: # letting 0.85 be our threshold here roughly based on lect. slides and trial&error
            matches[match_counter,0] = i # note which im1 feature was matched
            matches[match_counter,1] = argsorted[i,0] # use the argsorted matrix to find the corresponding im2 feature
            match_counter = match_counter + 1
    if matches.shape[0] > match_counter:
        matches = np.delete(matches, slice(match_counter, matches.shape[0]), axis=0)

    # TODO [COMPLETED*]: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!

    return matches

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
             residual, the error in the estimation of M given the point sets
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #

    A = np.zeros((2 * points3d.shape[0], 11))
    b = np.zeros((2* points2d.shape[0], 1))

    for i in range(points3d.shape[0]): # for every point, add the following rows to A and b
        A[2 * i] = [points3d[i,0], points3d[i,1], points3d[i,2], 1, 0, 0, 0, 0,
                     points3d[i,0] * points2d[i,0] * -1,
                     points3d[i,1] * points2d[i,0] * -1,
                     points3d[i,2] * points2d[i,0] * -1]
        A[2*i+1] = [0,0,0,0,points3d[i,0], points3d[i,1], points3d[i,2], 1,
                    points3d[i,0] * points2d[i,1] * -1,
                     points3d[i,1] * points2d[i,1] * -1,
                     points3d[i,2] * points2d[i,1] * -1]
        b[2*i] = [points2d[i,0]]
        b[2*i + 1] = [points2d[i,1]] # is this right?

    M, residual = np.linalg.lstsq(A,b,rcond=None)[:2]
    M = np.append(M, [1]) # add m_34 back into the projection matrix
    M = M.reshape(3,4)
    print('Performed least squares!')

    ########################
    # # Placeholder values. This M matrix came from a call to rand(3,4). It leads to a high residual.
    # print('Randomly setting matrix entries as a placeholder')
    # M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
    #               [0.6750, 0.3152, 0.1136, 0.0480],
    #               [0.1020, 0.1725, 0.7244, 0.9932]])
    # residual = 7 # Arbitrary stencil code initial value placeholder

    return M, residual

def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices.

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will transform a point into 
    a line within the second image - the epipolar line - such that F x' = l. 
    Fitting a fundamental matrix to a set of points will try to minimize the 
    error of all points x to their respective epipolar lines transformed 
    from x'. The residual can be computed as the difference from the known 
    geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the sum of the squared error in the estimation
    """
    ########################
    # TODO: Your code here #
    
    # First, fill A matrix with values of interest for Af = 0 
    A = np.zeros((points1.shape[0],9))
    for i in range(points1.shape[0]):
        x1 = points1[i,0]
        y1 = points1[i,1]
        x2 = points2[i,0]
        y2 = points2[i,1]
        A[i] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    # Now, perform SVD and extract the solution f for the aforementioned equation
    _,_,V = np.linalg.svd(A)
    solution_f = V[-1].reshape(3,3)

    # Next, correct the estimate of f by setting the rank to 2
    f_u, f_sigma, f_V = np.linalg.svd(solution_f)
    f_sigma[-1] = 0 # this sets the last row to 0, reducing the rank from 3 to 2!

    # After that, reconstruct the fundamental matrix with U sigma2 V
    F_matrix = np.dot(f_u, np.dot(np.diag(f_sigma), f_V))
    F_matrix /= np.linalg.norm(F_matrix) # Does this benefit from being normalized????

    # residual
    residual = 0
    for i in range(points1.shape[0]):
        x1 = points1[i,0]
        y1 = points1[i,1]
        x2 = points2[i,0]
        y2 = points2[i,1]
        homogenized_1 = np.array([x1, y1, 1])
        homogenized_2 = np.array([x2, y2, 1])
        residual += (np.dot(homogenized_2.T, np.dot(F_matrix, homogenized_1)))**2 # sum of squared error
    ########################

    # # Arbitrary intentionally incorrect Fundamental matrix placeholder
    # F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])
    # residual = 5 # Arbitrary stencil code initial value placeholder

    return F_matrix, residual

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Implement RANSAC to find the best fundamental matrix robustly
    by randomly sampling interest points. See the handout for a detailing of the RANSAC method.
    
    Inputs:
    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points across two images. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    Outputs:
    best_Fmatrix is the [3 x 3] fundamental matrix
    best_inliers1 and best_inliers2 are the [M x 2] subset of matches1 and matches2 that
    are inliners with respect to best_Fmatrix
    best_inlier_residual is the sum of the square error induced by best_Fmatrix upon the inlier set

    :return: best_Fmatrix, inliers1, inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    iterations = num_iters # change this as you like!
    size = 9 # 8 or 9 recommended
    inlier_threshold = 0.007 # change this to test out values!

    best_num_inliers = 0
    best_Fmatrix = np.zeros((3,3)) # Shape is 3x3 but values are unknown right now
    best_inliers_a = matches1[0:29, :] # RANDOM INITIALIZATION
    best_inliers_b = matches2[0:29, :] # RANDOM INITIALIZATION
    best_inlier_residual = 1e10 # large initialization

    for _ in range(iterations):
        # only use the subsets to calculate the fundamental matrix! Afterwards, use all the matches
        random_indices = np.random.randint(matches1.shape[0], size=size)
        matches1_subset = matches1[random_indices]
        matches2_subset = matches2[random_indices]

        # CHEAT FUNCTION (REPLACED IN STEP 4)
        # F_matrix, _ = cv2.findFundamentalMat(matches1_subset, matches2_subset, cv2.FM_8POINT, 1e10, 0, 1)
        F_matrix, residual = estimate_fundamental_matrix(matches1_subset, matches2_subset)

        # skip this iteration if the fundamental matrix is buggy (FOR CHEAT FUNCTION ONLY)
        if F_matrix is None:
            continue

        # Test agreement for all matches by checking how close x'T F x is to 0 (EDIT: IS THIS COVERED BY STEP 4?)
        homogeneous = np.ones((matches1.shape[0],1))
        m1_homogeneous = np.hstack((matches1, homogeneous))
        m2_homogeneous = np.hstack((matches2, homogeneous))

        intermediate_calculation = m1_homogeneous @ F_matrix # matmul
            # computes the values of x'.T * F * x and stores it in a vector
        xTFx_values = np.sum(intermediate_calculation * m2_homogeneous, axis=1)
        total_residual = np.sum(xTFx_values ** 2) # sum of squared error, as per autograder suggestion

        # compute inliers!
        inliers = np.abs(xTFx_values) < inlier_threshold # vector with True for inliers and False for outliers
        inliers1 = matches1[inliers]
        inliers2 = matches2[inliers]
        
        # if new best results, replace current best results
        if (inliers1.shape[0] > best_num_inliers) or (
            inliers1.shape[0] == best_num_inliers and
              total_residual < best_inlier_residual): # Score by fraction of correspondences that are inliers
                #if total_residual < best_inlier_residual: # Test: Scoring by total residual
            best_num_inliers = inliers1.shape[0] # number of elements in inliers1 vector
            print('New best number of inliers found: ' + str(best_num_inliers))

            best_Fmatrix = F_matrix
            best_inliers_a = inliers1
            best_inliers_b = inliers2
            best_inlier_residual = total_residual

        # append values to global variables
        inlier_counts.append(inliers1.shape[0])
        inlier_residuals.append(total_residual)
    ########################

    # Your RANSAC loop should contain a call to your 'estimate_fundamental_matrix()'

    # # Placeholder values
    # best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    # best_inliers_a = matches1[0:29, :]
    # best_inliers_b = matches2[0:29, :]
    # best_inlier_residual = 5 # Arbitrary stencil code initial value placeholder.
    print('best inlier residual was ' + str(best_inlier_residual))


    # For your report, we ask you to visualize RANSAC's 
    # convergence over iterations. 
    # For each iteration, append your inlier count and residual to the global variables:
    #   inlier_counts = []
    #   inlier_residuals = []
    # Then add flag --visualize-ransac to plot these using visualize_ransac()
    

    return best_Fmatrix, best_inliers_a, best_inliers_b, best_inlier_residual

#----------------------------------------------------------------------------------------#

image1_file = r"C:\Users\regen\Documents\Classes\fall_24\CS1430_Projects\cv-final-project\images\IMG_8966.jpeg"
image2_file = r"C:\Users\regen\Documents\Classes\fall_24\CS1430_Projects\cv-final-project\images\IMG_8967.jpeg"
test_stereo_matching(image1_file, image2_file)