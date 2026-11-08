

import numpy as np
from ..utils.constant import get_record_detail,get_placement_optimization,get_local_width, get_ignore_pixel_value, get_stabalize_method, get_dir
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from skimage.measure import regionprops
from ..utils.region_memo import occupied_bounds
INTERLOCKING_PIXEL = 3
CANDIDATE_CHUNK = 256  # candidates summed at once; bounds the memory of a large stone with many candidates
WALL_MAPS_KEPT = 4  # walls remembered: a step evaluates the wall and its mirror image
WALL_MAPS = {}
NUMBER_OF_PIXELS_AS_EPSILON = 3

def get_void_to_left(base,stone):
    """Empty pixels between the stone and the wall to its left and below it, over the size of the image.

    Only the rows and columns the stone occupies count, so the work stays inside them.
    """
    m_shape = base.matrix.shape
    rows, cols = np.nonzero(stone.matrix)
    if rows.size == 0:
        return np.float64(0)/(m_shape[1]*stone.height+m_shape[0]*stone.width)
    first_row, first_col = rows[0], cols.min()
    pixels = stone.matrix[first_row:rows[-1]+1,first_col:cols.max()+1] != 0

    #distance to the left
    filled = _placed_pixels(base.matrix[first_row:rows[-1]+1,:int(stone.center[0])])
    distance_l = (first_col+np.argmax(pixels,axis=1)+1-_last_filled(filled,axis=1)-1)[pixels.any(axis=1)]

    #distance to the floor
    filled = _placed_pixels(base.matrix[:int(stone.center[1]),first_col:cols.max()+1])
    distance_b = (first_row+np.argmax(pixels,axis=0)+1-_last_filled(filled,axis=0)-1)[pixels.any(axis=0)]

    positive_sum = float(distance_l[distance_l>0].sum())+float(distance_b[distance_b>0].sum())
    return np.float64(positive_sum)/(m_shape[1]*stone.height+m_shape[0]*stone.width)

def _placed_pixels(matrix):
    return (matrix != get_ignore_pixel_value()) & (matrix != 0)

def _last_filled(filled, axis):
    """Along axis, 1 + the index of the last True, or 0 without one."""
    if filled.shape[axis] == 0:
        return np.zeros(filled.shape[1-axis], dtype=np.int64)
    last = filled.shape[axis]-np.argmax(np.flip(filled,axis=axis),axis=axis)
    return np.where(filled.any(axis=axis),last,0)

def center_void(base, stone):
    """The third value of evaluate, from the window it counts in: -1 for a stone that overlaps the wall or has no pixel."""
    rows, cols = np.nonzero(stone.matrix)
    if rows.size == 0:
        return -1
    row_box, col_box = slice(rows[0],rows[-1]+1), slice(cols.min(),cols.max()+1)
    if np.any(np.multiply(base.matrix[row_box,col_box],stone.matrix[row_box,col_box])):
        return -1
    local_width = get_local_width()
    if local_width == 0:
        local_width_w = stone.height
        local_width_h = stone.width
    else:
        local_width_w = local_width
        local_width_h = local_width
    window = (slice(max(0, int(stone.center[1]-stone.height/2-local_width_h)),min(stone.matrix.shape[0], int(stone.center[1] + stone.height/2))),
              slice(max(0, int(stone.center[0]-stone.width/2-local_width_w/2)),min(stone.matrix.shape[1], int(stone.center[0]+stone.width/2+local_width_w/2))))
    nb_void_center = np.count_nonzero(base.matrix[window]+stone.matrix[window] == 0)
    return -(nb_void_center/((stone.height+local_width_h)*(stone.width+local_width_w)-stone.height*stone.width))

def evaluate(base, stone):
    """Evaluate a position of a stone

    :param base: Current base with placed stones
    :type base: Stonepacker2D.base.Base
    :param stone: Stone placed on the position
    :type stone: Stonepacker2D.stone.Stone
    :return: Distance to be minimized
    :rtype: float
    """
    write_txt = get_record_detail()['RECORD_PLACEMENT_VALUE']
    # check penetration
    product = np.multiply(base.matrix, stone.matrix)
    # plt.clf()
    # plt.imshow(stone.matrix)
    # plt.show()
    # plt.clf()
    # plt.imshow(base.matrix)
    # plt.show()
    # plt.clf()

    if np.any(product):
        # write to txt
        if write_txt == True:
            with open(get_dir()+f"record/place_ite{len(base.placed_bricks)}_brick{stone.id}_{stone.width}_{stone.rotate_from_ori}.txt", "a+") as f:
                f.write("{};{};{};{};{};{}\n".format(stone.center[0], stone.center[1],
                        100, 0, 0,0))
        return float("+inf"),-1,-1
    if not np.any(stone.matrix):
        if write_txt == True:
            with open(get_dir()+f"record/place_ite{len(base.placed_bricks)}_brick{stone.id}_{stone.width}_{stone.rotate_from_ori}.txt", "a+") as f:
                f.write("{};{};{};{};{};{}\n".format(stone.center[0], stone.center[1],
                        100, 0, 0,0))
        return float("+inf"),-1,-1
    distance = 0
    # _____________________________________________________________use local void
    _base_after = base.matrix+stone.matrix
    local_width = get_local_width()
    if local_width == 0:
        local_width_w = stone.height
        local_width_h = stone.width
    else:
        local_width_w = local_width
        local_width_h = local_width
    if get_placement_optimization() != "image_convolve":        
        local = _base_after[max(0, int(stone.center[1]-stone.height/2-local_width_h)):min(stone.matrix.shape[0], int(stone.center[1] + stone.height/2)),
                            max(0, int(stone.center[0]-stone.width/2-local_width_w)):min(stone.matrix.shape[1], int(stone.center[0]+stone.width/2))]

        nb_void = np.count_nonzero(local == 0)
        void_ratio = nb_void/((stone.height+local_width_h)*(stone.width+local_width_w)-stone.height*stone.width)
    else:
        m_shape = base.matrix.shape
        #distance to the left
        column_index = np.tile(np.arange(m_shape[1])+1,(m_shape[0],1))
        base_matrix_effective = np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0)
        column_index_base  = np.where(base_matrix_effective!=0,column_index,0)
        column_index_stone = np.where(stone.matrix!=0,column_index,np.inf)
        mask_l = np.zeros(m_shape)
        mask_l[:,0:int(stone.center[0])] = 1
        column_index_base_masked = np.multiply(column_index_base,mask_l)
        base_rb = np.max(column_index_base_masked,axis = 1)
        stone_lb = np.min(column_index_stone,axis = 1)
        distance_l = stone_lb-base_rb-1
        # print("base_rb",base_rb)
        # print("stone lb",stone_lb)
        # print("distance",distance_l)
        distance_positive_sum_l = np.sum(distance_l[(distance_l>0)&(distance_l<np.inf)])

        #distance to the floor
        row_index = np.tile(np.arange(m_shape[0])+1,(m_shape[1],1)).T
        row_index_base  = np.where(base_matrix_effective!=0,row_index,0)
        row_index_stone = np.where(stone.matrix!=0,row_index,np.inf)
        mask_b = np.zeros(m_shape)
        mask_b[0:int(stone.center[1]),:] = 1
        row_index_base_masked = np.multiply(row_index_base,mask_b)
        base_tb = np.max(row_index_base_masked,axis = 0)
        stone_bb = np.min(row_index_stone,axis = 0)
        distance_b = stone_bb-base_tb-1
        distance_positive_sum_b = np.sum(distance_b[(distance_b>0)&(distance_b<np.inf)])

        #void_ratio = 0.5*distance_positive_sum_l/(m_shape[1]*stone.height)+0.5*distance_positive_sum_b/(m_shape[0]*stone.width)
        void_ratio = (distance_positive_sum_l+distance_positive_sum_b)/(m_shape[1]*stone.height+m_shape[0]*stone.width)

    distance += void_ratio
    # _____________________________________________________________use local void where stone is placed in the center
    local_center = _base_after[max(0, int(stone.center[1]-stone.height/2-local_width_h)):min(stone.matrix.shape[0], int(stone.center[1] + stone.height/2)),
                        max(0, int(stone.center[0]-stone.width/2-local_width_w/2)):min(stone.matrix.shape[1], int(stone.center[0]+stone.width/2+local_width_w/2))]
    nb_void_center = np.count_nonzero(local_center == 0)
    void_ratio_center = nb_void_center/((stone.height+local_width_h)*(stone.width+local_width_w)-stone.height*stone.width)
    # --------------------------------------------------------------------------global density
    found, first_row, last_row, first_col = occupied_bounds(base)
    bbox = np.zeros(base.matrix.shape)
    if found:
        bbox[first_row:last_row+1,
             first_col:base.matrix.shape[1]] = 1
    else:
        bbox[0:3, :] = 1
    distance_gd = np.count_nonzero(np.multiply(bbox, stone.matrix)
                                   )/np.count_nonzero(bbox+stone.matrix)
    distance_gd = distance_gd / \
        (np.count_nonzero(stone.matrix)/np.count_nonzero(bbox))
    if distance_gd > 0.7:
        distance_gd *= 1
    else:
        distance_gd *= 0
    distance -= distance_gd
    # --------------------------------------------------------------------------x distance
    
    x_distance = stone.center[0]/stone.matrix.shape[1]
    #if get_placement_optimization() != "image_convolve": 
    distance += x_distance
    # write to txt
    if write_txt == True:
        with open(get_dir()+f"record/place_ite{len(base.placed_bricks)}_brick{stone.id}_{stone.width}_{stone.rotate_from_ori}.txt", "a+") as f:
            f.write("{};{};{};{};{};{}\n".format(stone.center[0], stone.center[1],
                    distance, void_ratio, -distance_gd,x_distance))
    return distance,-void_ratio,-void_ratio_center


def get_distance_to_interlocking(matrix):
    # revert 0 and 1
    matrix_boundary = np.where(matrix==0,1,0)
   
    # check if there is any zero pixel
    if len(np.argwhere(matrix_boundary==0))==0:
        return np.ones_like(matrix_boundary)*0.12
    # compute the distance to the boundary
    distance_matrix = scipy.ndimage.distance_transform_edt(matrix_boundary, return_distances=True, return_indices=False)
    return distance_matrix

import cv2
import math
def compute_edge(matrix):
    kernel = np.ones((1,3),np.int8)
    #the second half of the kernel is -1
    kernel[:,1] = 0
    kernel[:,2] = -1
    #convolve
    edge = ndimage.convolve(
        matrix, kernel, mode='nearest')
    edge = np.where(edge!=0,1,0)
    return edge

def get_distance_to_edge(matrix):   
    # plt.imshow(abs_sobel64f)
    # plt.show()
    matrix_boundary = compute_edge(matrix)
    # plt.imshow(matrix_boundary)
    # plt.show()
    # erosion and dilation (cv2 closing not working as expected)
    kernel = np.ones((NUMBER_OF_PIXELS_AS_EPSILON,1),np.uint8)
    matrix_boundary = cv2.erode(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    matrix_boundary = cv2.dilate(matrix_boundary.astype('uint8'),kernel,iterations = 1)
    # revert 0 and 1
    matrix_boundary = np.where(matrix_boundary==0,1,0)
   
    # check if there is any zero pixel
    if len(np.argwhere(matrix_boundary==0))==0:
        return np.ones_like(matrix_boundary)*0.12
    # compute the distance to the boundary
    distance_matrix = scipy.ndimage.distance_transform_edt(matrix_boundary, return_distances=True, return_indices=False)
    return distance_matrix


def get_phi_distance(base):
    #ground_pixel_value = get_ignore_pixel_value()
    #base = np.where(base==ground_pixel_value,0,base)
    # compute the minimal distance
    base_reverted  = np.where(base!=0,0,1)
    new_phi = scipy.ndimage.distance_transform_edt(base_reverted, return_distances=True,
                                         return_indices=False)
    return new_phi

def get_height(matrix):
    # matrix where the last row in dimension 0 is 0, the others are 1
    height_matrix = np.ones_like(matrix)
    height_matrix[0,:] = 0
    height_matrix = scipy.ndimage.distance_transform_edt(height_matrix, return_distances=True, return_indices=False)
    # normalize
    height_matrix = height_matrix/np.max(height_matrix)
    return height_matrix

def get_convolution_metric(base_phi, brick_bounding_box_matrix_mirrow):
    shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,
               (brick_bounding_box_matrix_mirrow.shape[1]-1) // 2]
    proximity_metric = ndimage.convolve(
       base_phi, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to)
    #proximity_metric = np.multiply(base_phi, stone)
    return proximity_metric

def stone_pixels(brick_bounding_box_matrix_mirrow):
    """Row and column offsets of the unmirrored stone's pixels, in the order ndimage.convolve visits them."""
    return np.nonzero(np.flip(brick_bounding_box_matrix_mirrow))

def candidate_metric(matrix, pixel_rows, pixel_cols, rows, cols):
    """get_convolution_metric at (rows, cols) only, where the stone pixels lie inside the image.

    ndimage.convolve adds the stone's pixels one by one from left to right; the running sum here keeps that order, so
    every value has the same bits as in the full image.
    """
    metric = np.empty(len(rows))
    for first in range(0, len(rows), CANDIDATE_CHUNK):
        last = first + CANDIDATE_CHUNK
        pixels = matrix[rows[first:last, None] + pixel_rows, cols[first:last, None] + pixel_cols]
        metric[first:last] = np.add.accumulate(pixels, axis=1)[:, -1]
    return metric

def column_window_min(matrix, first_row, length, column, cval, box):
    """Inside box, at each (r, c), the minimum of matrix[r + first_row : r + first_row + length, c + column]; cval outside the image.

    box is (first row, end row, first column, end column). Equals ndimage.minimum_filter with a footprint of one column:
    both take the minimum of the same values.
    """
    top, bottom, left, right = box
    read = np.full((bottom - top + length - 1, right - left), cval, dtype=matrix.dtype)
    start_row, start_col = top + first_row, left + column
    from_row, to_row = max(start_row, 0), min(start_row + read.shape[0], matrix.shape[0])
    from_col, to_col = max(start_col, 0), min(start_col + read.shape[1], matrix.shape[1])
    if from_row < to_row and from_col < to_col:
        read[from_row - start_row:to_row - start_row, from_col - start_col:to_col - start_col] = matrix[from_row:to_row, from_col:to_col]
    filtered = ndimage.minimum_filter1d(read, length, axis=0, mode='constant', cval=cval)
    return filtered[length // 2:length // 2 + bottom - top]

def convolve_nonzero(matrix, mirrored_kernel, cval):
    """Where ndimage.convolve with origin (size-1)//2 is nonzero, for non-negative float matrix and binary kernel.

    The kernel then sits with its unmirrored top-left at each pixel, so only the bottom and right can leave the image.
    """
    kernel = np.flip(mirrored_kernel).astype(np.float64)
    occupied = (matrix != 0).astype(np.uint8)
    if cval != 0:
        occupied = cv2.copyMakeBorder(occupied, 0, kernel.shape[0] - 1, 0, kernel.shape[1] - 1, cv2.BORDER_CONSTANT, value=1)
    overlap = cv2.filter2D(occupied, cv2.CV_64F, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    return overlap[:matrix.shape[0], :matrix.shape[1]] > 0.5

def wall_maps(base):
    """The images evaluate_convolve reads that depend on the wall alone, computed once per wall in this process.

    A pose task unpickles its own copy of the wall, so the wall is recognised by its content. The images are read-only.
    """
    left_tops, right_tops = np.asarray(base.rock_left_tops), np.asarray(base.rock_right_tops)
    key = (
        get_ignore_pixel_value(),
        base.matrix.dtype.str,
        base.matrix.shape,
        base.matrix.tobytes(),
        left_tops.shape,
        left_tops.tobytes(),
        right_tops.shape,
        right_tops.tobytes(),
    )
    maps = WALL_MAPS.get(key)
    if maps is None:
        wall_with_no_bound = np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0)
        contour_dilated = cv2.dilate(wall_with_no_bound, np.ones((3, 3), np.uint8), anchor=(1, 1), iterations=1)
        interlocking_point_matrix = np.zeros(base.matrix.shape)
        # make all top points of placed stones 1
        if left_tops.shape[0] >= 1:
            interlocking_point_matrix[left_tops[:, 1], left_tops[:, 0]] = 1
        if right_tops.shape[0] >= 1:
            interlocking_point_matrix[right_tops[:, 1], right_tops[:, 0]] = 1
        interlocking_distance = get_distance_to_interlocking(
            interlocking_point_matrix
        )  # maximize the distance to the interlocking
        maps = (
            contour_dilated,
            interlocking_distance,
            get_distance_to_edge(base.matrix),
            get_phi_distance(base.matrix),
            get_height(base.matrix),
        )
        for image in maps:
            image.flags.writeable = False
        if len(WALL_MAPS) >= WALL_MAPS_KEPT:
            WALL_MAPS.clear()
        WALL_MAPS[key] = maps
    return maps


def evaluate_convolve(base,brick,brick_bounding_box_matrix_mirrow=None,region_potential=None):
    stone_width = brick.width
    stone_height = brick.height
    stone_center = (brick.width/2,brick.height/2)
    # the minimum filters' windows: their footprint's centre is the middle pixel, or the one after it in an even dimension
    bbox_rows, bbox_cols = brick_bounding_box_matrix_mirrow.shape
    even_rows, even_cols = 1 - bbox_rows % 2, 1 - bbox_cols % 2
    touch_start = math.ceil(stone_center[1])

    # find positions that have support from both left and right
    contour_dilated, interlocking_distance, touch_distance, base_phi, base_height = wall_maps(base)

    #dilate the region to filter out floating positions
    # kernel_dilation_y = np.ones((2, 1), np.uint8)
    # contour_dilated = cv2.dilate(base.matrix, kernel_dilation_y, anchor=(0,1),iterations=1)

    # everything below is read only where region_potential is set: the images hold that box of the wall
    potential_rows, potential_cols = np.nonzero(region_potential)
    box = (potential_rows[0], potential_rows[-1] + 1, potential_cols.min(), potential_cols.max() + 1)
    in_box = (slice(box[0], box[1]), slice(box[2], box[3]))
    box_potential = region_potential[in_box]
    contour_box = contour_dilated[box[0]:box[1] + bbox_rows - 1, box[2]:box[3] + bbox_cols - 1]

    # left contact
    left_mask = np.zeros_like(brick_bounding_box_matrix_mirrow)
    # print(brick_bounding_box_matrix_mirrow.shape)
    # print(math.floor(stone_center[0]))
    left_mask[:,0:math.floor(stone_center[0])-2] = 1
    left_brick_bounding_box_matrix_mirrow = brick_bounding_box_matrix_mirrow*left_mask
    left_overlap_contour_mask = np.where(convolve_nonzero(
        contour_box, left_brick_bounding_box_matrix_mirrow, cval=0.0)[:box_potential.shape[0], :box_potential.shape[1]] != 0, 1, 0)
    #right contact
    right_mask = np.zeros_like(brick_bounding_box_matrix_mirrow)
    right_mask[:,math.ceil(stone_center[0])+2:] = 1
    right_brick_bounding_box_matrix_mirrow = brick_bounding_box_matrix_mirrow*right_mask
    right_overlap_contour_mask = np.where(convolve_nonzero(
        contour_box, right_brick_bounding_box_matrix_mirrow, cval=0.0)[:box_potential.shape[0], :box_potential.shape[1]] != 0, 1, 0)

    stable_potential = np.multiply(right_overlap_contour_mask, left_overlap_contour_mask)
    if not np.any(np.multiply(box_potential, stable_potential)):
        #print("No stable potential")
        stable_potential = np.ones_like(box_potential)
    #stable_potential = np.ones_like(region_potential)
    box_potential = np.multiply(box_potential,stable_potential)

    # find locations matches mason's practice
    #left interlocking distance map and convolve with the bounding of stone
    left_interlocking_distance_convolved = column_window_min(interlocking_distance, -even_rows, 1, -even_cols, np.inf, box)
    left_interlocking_distance_convolved[left_interlocking_distance_convolved==0.12] = np.inf
    #right interlocking distance map and convolve with the bounding of stone
    right_interlocking_distance_convolved = column_window_min(interlocking_distance, -even_rows, 1, bbox_cols - 1 - even_cols, np.inf, box)
    right_interlocking_distance_convolved[right_interlocking_distance_convolved==0.12] = np.inf
    
    #touch

    left_touch_distance_convolved = column_window_min(touch_distance, touch_start - even_rows, bbox_rows - touch_start, -even_cols, 0.0, box)
    #right interlocking distance map and convolve with the bounding of stone
    right_touch_distance_convolved = column_window_min(touch_distance, touch_start - even_rows, bbox_rows - touch_start, bbox_cols - 1 - even_cols, 0.0, box)
    

    interlocking_thre = min(INTERLOCKING_PIXEL,stone_width/2)
    interlocking_thre = min(interlocking_thre,stone_height)
   
    #mason_criteria = np.where((left_interlocking_distance_convolved>=interlocking_thre)&(right_interlocking_distance_convolved>=interlocking_thre),1,0)
    mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre)&(right_touch_distance_convolved<=1))|\
    ((right_interlocking_distance_convolved>=interlocking_thre)&(left_touch_distance_convolved<=1))|\
        ((right_touch_distance_convolved<=1)&(left_touch_distance_convolved<=1)),1,0)
    

    iteration_ = 1
    while not np.any(np.multiply(box_potential,mason_criteria)) and iteration_<4:

        # mason_criteria = np.where((left_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))\
        #                           &(right_interlocking_distance_convolved>=(interlocking_thre/(iteration_+1))))
        mason_criteria = np.where(((left_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))&(right_touch_distance_convolved<=1))|\
        ((right_interlocking_distance_convolved>=interlocking_thre/(iteration_+1))&(left_touch_distance_convolved<=1))|\
        ((right_touch_distance_convolved<=1)&(left_touch_distance_convolved<=1)),1,0)
        iteration_+=1
    if not np.any(np.multiply(box_potential,mason_criteria)):
        #print("No interlocking positions")
        mason_criteria = np.ones_like(box_potential)

    # plt.imshow(interlocking_point_matrix)
    # plt.title("mason criterial with interlocking thre = {}".format(interlocking_thre))
    # plt.show()
    weight_height = np.sqrt(stone_width**2+stone_height**2)*1000
    # only the candidates are read: their sums replace the full images when every stone pixel is inside the image
    candidates = (box_potential != 0) & (mason_criteria!=0)
    candidate_rows, candidate_cols = np.nonzero(candidates)
    candidate_rows, candidate_cols = candidate_rows + box[0], candidate_cols + box[2]
    pixel_rows, pixel_cols = stone_pixels(brick_bounding_box_matrix_mirrow)
    if candidate_rows.size and candidate_rows.max() + pixel_rows.max() < base.matrix.shape[0] \
            and candidate_cols.max() + pixel_cols.max() < base.matrix.shape[1]:
        candidate_phi = candidate_metric(base_phi, pixel_rows, pixel_cols, candidate_rows, candidate_cols)#smaller better
        candidate_height = candidate_metric(base_height*weight_height, pixel_rows, pixel_cols, candidate_rows, candidate_cols)
        candidate_distance = candidate_phi+candidate_height#smaller better
        best_candidate = np.argmin(candidate_distance)#the first of equal scores, as np.argwhere gives
        best_loc = np.array([candidate_rows[best_candidate], candidate_cols[best_candidate]])
        best_distance_optimization = candidate_distance[best_candidate]
        best_phi = -candidate_phi[best_candidate]
    else:
        score_optimization_phi = get_convolution_metric(base_phi, brick_bounding_box_matrix_mirrow)#smaller better
        score_optimization_height = get_convolution_metric(base_height*weight_height, brick_bounding_box_matrix_mirrow)
        distance_optimization = score_optimization_phi+score_optimization_height#smaller better
        whole_image = np.zeros(base.matrix.shape, dtype=bool)
        whole_image[in_box] = candidates
        score_potential = np.where(whole_image, distance_optimization, float("+inf"))
        best_score = np.min(score_potential)
        best_loc = np.argwhere(score_potential == best_score)[0]
        # get the proximity metric of the best location
        best_distance_optimization = distance_optimization[best_loc[0],best_loc[1]]
        best_phi = -score_optimization_phi[best_loc[0],best_loc[1]]

    # print("Best location",best_loc)
    # print("Best left interlocking",left_interlocking_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best right interlocking",right_interlocking_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best left touch",left_touch_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best right touch",right_touch_distance_convolved[best_loc[0],best_loc[1]])
    # print("Best height distance",score_optimization_height[best_loc[0],best_loc[1]])
    # print("Best void distance",-best_phi)

    
    
    #return best_loc,best_distance_optimization,best_phi,left_touch_distance_convolved[best_loc[0],best_loc[1]],right_touch_distance_convolved[best_loc[0],best_loc[1]]
    best_in_box = (best_loc[0] - box[0], best_loc[1] - box[2])
    return {"best_loc":best_loc,"best_distance":best_distance_optimization,"best_phi":best_phi,"best_left_interlocking":left_interlocking_distance_convolved[best_in_box],\
            "best_right_interlocking":right_interlocking_distance_convolved[best_in_box],"best_left_touch":left_touch_distance_convolved[best_in_box],\
                "best_right_touch":right_touch_distance_convolved[best_in_box]}