#Importing necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from tqdm import *

#Function to perform preprocessing operations on the images
def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.GaussianBlur(gray, (15,15), 0)
    return processed_img

#Function to find features in the two imagess
def feature_detection(img1, img2):
    processed_img1, processed_img2 = preprocessing(img1), preprocessing(img2)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(processed_img1, None)
    kp2, desc2 = sift.detectAndCompute(processed_img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    #Extracting good matches
    good = []
    for m, n in matches:
        if m.distance < 0.2*n.distance:
            good.append([m])
    
    left_pts  = np.float32([kp1[m[0].queryIdx].pt for m in good])
    right_pts  = np.float32([kp2[m[0].trainIdx].pt for m in good])
    return left_pts, right_pts

#Function to compute Fundamental Matrix given feature points in two images
def fundamental_matrix(left_pts, right_pts):
    A = []
    
    for i in range(8):
        src_x, src_y  = left_pts[i]
        dst_x, dst_y = right_pts[i]
        A_row = np.asarray([dst_x*src_x, dst_x*src_y, dst_x, dst_y*src_x, dst_y*src_y, dst_y, src_x, src_y, 1])
        A.append(A_row)
        
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)
    F = np.reshape(V[-1, :], (3, 3))
    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0
    F = u @ (s @ vt)
    F = F / F[-1,-1]
    F = (abs(F) > (10**(-3))) * F
    return F

#Ransac function to find the best Fundamental Matrix that satisfies the epipolar constraint
def ransac(src_pts, dst_pts, iterations, threshold):
    number_of_points = len(src_pts)
    max_inlier_count = 0
    
    for i in range(iterations):
        inlier = 0
        random_indices = np.random.choice(number_of_points, size = 8, replace=False)
        random_src_pts = src_pts[random_indices, :]
        random_dst_pts = dst_pts[random_indices, :]
        
        F = fundamental_matrix(random_src_pts, random_dst_pts)

        for i in range(number_of_points):
            a, b = dst_pts[i], src_pts[i]
            a, b = np.append(a, 1).reshape((3, 1)), np.append(b, 1).reshape((3, 1))
            c = (a.T @ F) @ b
            
            if abs(c) <= threshold:
                inlier+=1
            
        if inlier > max_inlier_count:
            max_inlier_count = inlier
            best_F = F          
    return best_F

#Function to compute essential matrix
def essential_matrix(F, K1, K2):
    E = (K2.T @ F) @ K1
    U, _, V = np.linalg.svd(E)
    S = [1, 1, 0]
    S = np.diag(S)
    E = np.matmul(np.matmul(U, S), V)
    return E

#Function to decompose Essential Matrix to find Rotation and Translation
def decompose_essential_matrix(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, V = np.linalg.svd(E)
    
    R_set = []
    C_set = []
    
    R_set.append((U @ W) @ V)
    R_set.append((U @ W) @ V)
    R_set.append((U @ W.T) @ V)
    R_set.append((U @ W.T) @ V)
    C_set.append(U[:, 2])
    C_set.append(-U[:, 2])
    C_set.append(U[:, 2])
    C_set.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R_set[i]) < 0):
            R_set[i] = -R_set[i]
            C_set[i] = -C_set[i]

    return R_set, C_set

#Function to find the Projection Matrix
def projection_matrix(K, R, C):
    I  = np.identity(3)
    C = np.reshape(C, (3, 1))
    return (K @ (R @ np.hstack((I, -C))))

#Function to find 3D point given 2D pixel locations
def linear_triangulation(R_set, C_set, left_pts, right_pts, K1, K2):
    x3D_set = []

    for i in range(len(R_set)):
        R1, R2 = np.identity(3), R_set[i]
        C1, C2 = np.zeros((3, 1)),  C_set[i].reshape(3,1)
        
        P1 = projection_matrix(K1, R1, C1)
        P2 = projection_matrix(K2, R2, C2)
        
        p1, p2, p3 = P1
        p1_, p2_, p3_ = P2
        
        p1, p2, p3 =  p1.reshape(1,-1), p2.reshape(1,-1), p3.reshape(1,-1) 
        p1_, p2_, p3_ =  p1_.reshape(1,-1), p2_.reshape(1,-1), p3_.reshape(1,-1) 
        
        x3D =[]
        
        for left_pt, right_pt in zip(left_pts, right_pts):
            x, y = left_pt
            x_, y_ = right_pt
            A = np.vstack((y * p3 - p2, p1 - x * p3, y_ * p3_ - p2_, p1_-x_ * p3_ ))
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[-1]
            x3D.append(X[:3])
         
        x3D_set.append(x3D)
    return x3D_set

#Function to find best camera pose configuration out of the four options
def disambiguate_camera_pose(R_set, C_set, x3D_set):
    best_i = 0
    max_positive_depths = 0
    
    for i in range(len(R_set)):
        n_positive_depths=  0
        R, C = R_set[i],  C_set[i].reshape(-1, 1) 
        r3 = R[2].reshape(1, -1)
        x3D = x3D_set[i]
        
        for X in x3D:
            X = X.reshape(-1, 1) 
            if r3 @ (X - C) > 0 and X[2] > 0: 
                n_positive_depths += 1
                
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths 

    R, C = R_set[best_i], C_set[best_i]
    return R, C

#Function to find start and end co-ordinates given equation of an epiline
def epiline_coordinates(lines, img):
    lines = lines.reshape(-1, 3)
    c = img.shape[1]
    co_ordinates = []
    
    for line in lines:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        co_ordinates.append([[x0, y0], [x1, y1]])
        
    return co_ordinates

#Function to draw epilines on the two corresponding images
def draw_epilines(l_epiline_coords, r_epiline_coords, left_pts, right_pts, img1, img2):
    img1_copy = copy.deepcopy(img1)
    img2_copy = copy.deepcopy(img2)

    for l_epiline_coord, r_epiline_coord, left_pt, right_pt in zip(l_epiline_coords, r_epiline_coords, np.int32(left_pts), np.int32(right_pts)):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1_copy = cv2.line(img1_copy, l_epiline_coord[0], l_epiline_coord[1], color, 2)
        img1_copy = cv2.circle(img1_copy, (left_pt[0][0], left_pt[0][1]), 7, color, -1)
        img2_copy = cv2.line(img2_copy, r_epiline_coord[0], r_epiline_coord[1], color, 2)
        img2_copy = cv2.circle(img2_copy, (right_pt[0][0], right_pt[0][1]), 7, color, -1)
    
    return img1_copy, img2_copy

#Function to find co-ordinates of epilines after applying perspective transformation
def warped_epiline_coords(l_epiline_coords, r_epiline_coords, H1, H2):
    
    l_epiline_coords = np.array(l_epiline_coords, dtype = np.float32)
    r_epiline_coords = np.array(r_epiline_coords, dtype = np.float32)
    
    l_epiline_start_coords = l_epiline_coords[:, 0].reshape(-1,1,2)
    l_epiline_end_coors = l_epiline_coords[:, 1].reshape(-1,1,2)
    
    r_epiline_start_coords = r_epiline_coords[:, 0].reshape(-1,1,2)
    r_epiline_end_coords = r_epiline_coords[:, 1].reshape(-1,1,2)

    warped_l_epiline_start_coords = cv2.perspectiveTransform(l_epiline_start_coords, H1).squeeze()
    warped_l_epiline_end_coords = cv2.perspectiveTransform(l_epiline_end_coors, H1).squeeze()

    warped_r_epiline_start_coords = cv2.perspectiveTransform(r_epiline_start_coords, H2).squeeze()
    warped_r_epiline_end_coords = cv2.perspectiveTransform(r_epiline_end_coords, H2).squeeze()

    warped_l_epiline_coords = []
    warped_r_epiline_coords = []
    
    for start, end in zip(warped_l_epiline_start_coords, warped_l_epiline_end_coords):
        start, end = start.astype(int), end.astype(int)
        warped_l_epiline_coords.append([start, end])
    
    for start, end in zip(warped_r_epiline_start_coords, warped_r_epiline_end_coords):
        start, end = start.astype(int), end.astype(int)
        warped_r_epiline_coords.append([start, end])

    return warped_l_epiline_coords, warped_r_epiline_coords

#Function to calcalate sum of absolte differences
def sum_of_abs_diff(pixel_vals_1, pixel_vals_2):
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

#Function to find corresponding block in the other images using SAD
def compare_blocks(y, x, block_left, right_array, window, search_range):
    x_min = max(0, x - search_range)
    x_max = min(right_array.shape[1], x + search_range)
    first = True
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+window, x: x+window]
        sad = sum_of_abs_diff(block_left, block_right)
        
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index

img1 = cv2.imread("data/curule/im0.png")
img2 = cv2.imread("data/curule/im1.png")

left_pts, right_pts = feature_detection(img1, img2)

F = ransac(left_pts, right_pts, 1500, 0.02)

K1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
K2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
# K1 = np.array([[1742.11, 0, 804.9], [0, 1742.11, 541.22], [0, 0, 1]])
# K2 = np.array([[1742.11, 0, 804.9], [0, 1742.11, 541.22], [0, 0, 1]])
# K1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
# K2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
E = essential_matrix(F, K1, K2)

R_set, C_set = decompose_essential_matrix(E)
x3D_set = linear_triangulation(R_set, C_set, left_pts, right_pts, K1, K2)
R, T = disambiguate_camera_pose(R_set, C_set, x3D_set)
print(R)
print(T)

l_epilines = cv2.computeCorrespondEpilines(right_pts.reshape(-1, 1, 2), 2, F)
r_epilines = cv2.computeCorrespondEpilines(left_pts.reshape(-1, 1, 2), 1, F)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(left_pts), np.float32(right_pts), F, imgSize=(w1, h1))
print(H1)
print(H2)

l_epiline_coords = epiline_coordinates(l_epilines, img1)
r_epiline_coordinates = epiline_coordinates(r_epilines, img2)
a, b = draw_epilines(l_epiline_coords, r_epiline_coordinates, left_pts.reshape(-1, 1, 2), right_pts.reshape(-1, 1, 2), 
                     img1, img2)
out = np.hstack((a, b))

img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
out = np.hstack((img1_rectified, img2_rectified))

left_dst_pts = cv2.perspectiveTransform(left_pts.reshape(-1, 1, 2), H1).squeeze()
right_dst_pts = cv2.perspectiveTransform(right_pts.reshape(-1, 1, 2), H2).squeeze()

warped_l_epiline_coords, warped_r_epiline_coords = warped_epiline_coords(l_epiline_coords, r_epiline_coordinates, H1, H2)
a, b = draw_epilines(warped_l_epiline_coords, warped_r_epiline_coords, left_dst_pts.reshape(-1, 1, 2), 
                     right_dst_pts.reshape(-1, 1, 2), img1_rectified, img2_rectified)
out = np.hstack((a, b))
cv2.imwrite('b\rectified_epilines.png', out)

a_gray = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
b_gray = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

window = 7
search_range = 56

a_gray = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
b_gray = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
a_gray = a_gray.astype(float)
b_gray = b_gray.astype(float)
h, w = a_gray.shape

disparity_map = np.zeros((h, w))
BLOCK_SIZE = 7
SEARCH_BLOCK_SIZE = 56

for y in tqdm(range(window, h-window)):
    for x in range(window, w-window):
        block_left = a_gray[y:y + window, x:x + window]
        min_index = compare_blocks(y, x, block_left, b_gray, window, search_range) 
        disparity_map[y, x] = abs(min_index[1] - x)

disparity = np.uint8(disparity_map * 255 / np.max(disparity_map))
heatmap = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

f = K1[0,0]
baseline = 88.39
depth = (baseline * f) / (disparity + 1e-15)
depth[depth > 30000] = 30000
depth_map = np.uint8(depth * 255 / np.max(depth))
heatmap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
