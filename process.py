import numpy as np 
import cv2
import math
from scipy import ndimage
import copy
from keras import layers, models
import Sudoku_Solver

model = models.load_model("./models/model2.h5")
width , height = 450,450


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orginal_image = image.copy()
    gau_image = cv2.GaussianBlur(gray_image , (5,5) , 6)
    return gau_image

def thresholding(Gauss_image):
    threshold_image = cv2.adaptiveThreshold(Gauss_image , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV , 11 ,2)
    return threshold_image

def find_contours(threshold_image):
    Contours = None
    Contours , hier = cv2.findContours(threshold_image , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return Contours 

def find_biggest_contours(contours):
    max_area = 0
    biggest_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area and area > 100:
            max_area = area
            biggest_contour = c
                
    return biggest_contour , max_area

def get_coordinates(biggest_contour):
    no_of_corner = 4
    max_iter = 200
    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1
        epsilon = coefficient * cv2.arcLength(biggest_contour , True)
        poly_approx = cv2.approxPolyDP(biggest_contour, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == no_of_corner:
            return hull
        else:
            if len(hull) > no_of_corner:
                coefficient += .01
            else:
                coefficient -= .01
    return None

def get_points(hull):
    mypoints = hull.reshape((4,2))
    mynewpoints = np.zeros((4,1,2) , dtype=np.int32)
    add = mypoints.sum(1)
    mynewpoints[0] = mypoints[np.argmin(add)]
    mynewpoints[3] = mypoints[np.argmax(add)]

    diff = np.diff(mypoints , axis=1)

    mynewpoints[1] = mypoints[np.argmin(diff)]
    mynewpoints[2] = mypoints[np.argmax(diff)]
    return mynewpoints

def prepossessing_for_model(main_board):
    main_board = cv2.cvtColor(main_board, cv2.COLOR_BGR2GRAY)
    main_board = cv2.GaussianBlur(main_board, (5, 5), 2)
    main_board = cv2.adaptiveThreshold(main_board, 255, 1, 1, 11, 2)
    main_board = cv2.bitwise_not(main_board)
    _, main_board = cv2.threshold(main_board, 10, 255, cv2.THRESH_BINARY)
    return main_board

def digit_component(image):
    image = image.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) <= 1:
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    output_image = np.zeros(output.shape)
    output_image.fill(255)
    output_image[output == max_label] = 0
    return output_image


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

def prepare(img_array):
    new_array = img_array.reshape(-1, 32, 32, 1)
    new_array = new_array.astype(np.float32)
    new_array /= 255
    return new_array

def is_grid_equal(old_grid, new_grid):
    for row in range(9):
        for col in range(9):
            if old_grid[row][col] != new_grid[row][col]:
                return False
    return True

def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    unit_vector2 = unit_vector2.reshape(unit_vector2.shape[1] , unit_vector2.shape[0])
    dot_product = np.dot(unit_vector_1, unit_vector2)
    angle = np.arccos(dot_product)
    angle = angle * 57.2958
    return angle

def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon

def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest

def write_output_on_image(image, grid, user_grid):
    print("writing on image....")
    grid_size = 9
    width = image.shape[1] // grid_size
    height = image.shape[0] // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            if user_grid[i][j] != 0:
                continue

            text = str(grid[i][j])
            offset_x = width // 15
            offset_y = height // 15
            (text_height, text_width), baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            font_scale = 0.5 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + offset_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + offset_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    return image

def get_prediction(main_board):
    grid_dim = 9
    grid = []
    for i in range(grid_dim):
        row = []
        for j in range(grid_dim):
            row.append(0)
        grid.append(row)

    height = main_board.shape[0] // 9
    width = main_board.shape[1] // 9
    
    offset_width = math.floor(width / 10)
    offset_height = math.floor(height / 10)

    for i in range(grid_dim):
        for j in range(grid_dim):

            crop_image = main_board[height * i + offset_height:height * (i + 1) - offset_height, width * j + offset_width:width * (j + 1) - offset_width]

            ratio = 0.6

            while np.sum(crop_image[0]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]
                
            while np.sum(crop_image[:, -1]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)

            while np.sum(crop_image[:, 0]) <= (1 - ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)
                
            while np.sum(crop_image[-1]) <= (1 - ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]

            crop_image = cv2.bitwise_not(crop_image)
            crop_image = digit_component(crop_image)

            digit_pic_size = 32
            crop_image = cv2.resize(crop_image, (digit_pic_size, digit_pic_size))

            if crop_image.sum() >= digit_pic_size ** 2 * 255 - digit_pic_size * 1 * 255:
                grid[i][j] = 0
                continue

            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]

            if center_region.sum() >= center_width * center_height * 255 - 255:
                grid[i][j] = 0
                continue

            _, crop_image = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY)
            crop_image = crop_image.astype(np.uint8)
            
            crop_image = cv2.bitwise_not(crop_image)
            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image, shift_x, shift_y)
            crop_image = shifted
            crop_image = cv2.bitwise_not(crop_image)

            crop_image = prepare(crop_image)

            prediction = model.predict([crop_image])
            grid[i][j] = np.argmax(prediction[0])
    return grid


def main(image , solved_old_sudoku):
    
    processed_image = preprocess_image(image)
    
    threshold_image = thresholding(processed_image)
    
    Contours = find_contours(threshold_image)
    
    if Contours is None:
        return image
    
    biggest_contour = find_biggest_contours(Contours)
    
    if biggest_contour[0] is None:
        return image
    
    hull = get_coordinates(biggest_contour[0])
    
    if hull is None:
        return image
    
    new_points = get_points(hull)
    src = np.float32(new_points)
    dst = np.float32([[0, 0], [width, 0],[0, height], [width, height]])
    
    matrix = cv2.getPerspectiveTransform(src , dst)
    wrap_board = cv2.warpPerspective(image , matrix , (width,height)) 
    wrap_sudoku_image = copy.deepcopy(wrap_board)
    
    sudoku_board = prepossessing_for_model(wrap_board)
    grid  = get_prediction(sudoku_board)
    user_grid = copy.deepcopy(grid)
    
    
    if (solved_old_sudoku is not None) and is_grid_equal(solved_old_sudoku, grid):
        if Sudoku_Solver.check_for_non_zero_board(grid):
            wrap_sudoku_image = write_output_on_image(wrap_sudoku_image, solved_old_sudoku, user_grid)
    else:  
        Sudoku_Solver.solve_sudoku(grid)
        if Sudoku_Solver.check_for_non_zero_board(grid):
            wrap_sudoku_image = write_output_on_image(wrap_sudoku_image, grid, user_grid)
            solved_old_sudoku = copy.deepcopy(grid)

    print(user_grid)
    print("**********************************")
    print(grid)
    result_sudoku = cv2.warpPerspective(wrap_sudoku_image, matrix, (image.shape[1], image.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, image)
    return result 

