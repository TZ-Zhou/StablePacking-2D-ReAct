import matplotlib.pyplot as plt
import cv2
from copy import deepcopy

import numpy as np
from skimage.measure import label,regionprops_table,regionprops
from scipy import ndimage
from ..entity import Brick,Stone
from .place_stone import place_one_stone
from multiprocessing import Pool
import pickle
from ..utils.constant import get_record_detail,get_record_time,get_cut_id,get_wedge_id,set_stabalize_method,get_selection_weights, get_ignore_pixel_value, get_number_cpu, get_rotate_state, get_dilation, get_erosion, get_dump, get_dir, get_plot_placement, get_filter_policy, get_trace
from .selection import select, filter_log, write_trace, next_event, to_world, FEATURE_NAMES
from ..utils.logger import get_main_logger
from ..utils.plot import save_matrix_plot, save_matrix
from ..evaluate import evaluate_kine
import time
import random
import json
import os
import pandas as pd
sequential_run = False

def trace_event(meta, base, vendor, current_bricks, all_bricks, xs, ys, distances, features, weights, selection):
    """Snapshot of one selection decision (inputs + live result) for offline replay; returns its event id.

    x, y, rotation and wall are in the evaluation frame (np.fliplr for side='right');
    x_world and rotation_world are the same poses in the wall frame (selection.to_world).
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    event = next_event()
    rotation = np.arange(len(all_bricks)) // len(current_bricks) * 90
    width = np.array([b.width for b in all_bricks])
    x_world, rotation_world = to_world(meta['side'], np.array(xs), rotation, width, all_bricks[0]._w)
    meta = {**meta, 'run_id': get_trace(), 'event': event, 'policy': get_filter_policy(),
            'frame': 'mirrored' if meta['side'] == 'right' else 'world', 'world_width': all_bricks[0]._w,
            'branch': selection.branch, 'stages': [s.tolist() for s in selection.stages],
            'np_rng': {'pos': int(np_state[2]), 'has_gauss': int(np_state[3]), 'cached_gaussian': float(np_state[4])},
            'py_rng': {'version': py_state[0], 'gauss_next': py_state[2]}}
    write_trace(get_dir()+"trace", meta,
                features_raw=features, weights=np.asarray(weights, dtype=float),
                stone_id=np.array([b.id for b in all_bricks]), rotation=rotation, width=width,
                height=np.array([b.height for b in all_bricks]),
                x=np.array(xs), y=np.array(ys), x_world=x_world, rotation_world=rotation_world,
                distance=np.array(distances, dtype=float),
                wall=base.matrix, inventory_ids=np.array(list(vendor.stones.keys())),
                sampled_ids=np.array([b.id for b in current_bricks]),
                np_rng_keys=np_state[1], py_rng_state=np.array(py_state[1]),
                selected=np.array(-1 if selection.index is None else selection.index))
    return event

def trace_commit(record):
    """One line per sampling attempt in trace/commits.jsonl: joins its filter events to the outcome."""
    os.makedirs(get_dir()+"trace", exist_ok=True)
    with open(get_dir()+"trace/commits.jsonl", "a") as f:
        f.write(json.dumps(record, default=lambda v: v.item()) + "\n")

def get_cut_stone(base,min_area = 9, max_area = 100,max_width = 10,add_element_type = 'flexible'):
    # find the largest hole in the base
    nonoccupied = np.where(base.matrix==0,1,0)
    if not base.placed_rocks:
        return [],[],[]
    last_placed_stone = base.placed_rocks[-1]
    last_placed_stone_top = np.max(np.argwhere(last_placed_stone.matrix!=0),axis = 0)[0]
    not_exceeding_last_placed_stone_mask =np.zeros(base.matrix.shape)
    not_exceeding_last_placed_stone_mask[0:np.max(np.argwhere(last_placed_stone.matrix!=0),axis = 0)[0],:] = 1
    label_image = label(np.multiply(not_exceeding_last_placed_stone_mask,nonoccupied),background = 0,connectivity = 1)
    props = regionprops_table(label_image,properties = ['label','area','bbox'])
    data = pd.DataFrame(props)
    #find the row with area in range min_area and max_area
    target_data = data[(data['area']>=min_area) & (data['area']<=max_area) & (data['bbox-3']-data['bbox-1']<=max_width) &(data['bbox-2']==last_placed_stone_top)]
    target_labels = target_data['label'].values
    if len(target_labels) == 0:
        return [],[],[]
    else:
        cubes = []
        cubes_xs = []
        cubes_ys = []
        if add_element_type == 'cube':
            #using cubes
            for target_label in target_labels:
                target_hole = np.where(label_image==target_label,1,0)
                cube_size = 3
                while True:
                    kernel=np.ones((cube_size,cube_size))
                    eroded = ndimage.binary_erosion(target_hole,structure = kernel).astype(target_hole.dtype)
                    
                    if eroded.sum() == 0:
                        break
                    else:
                        cube_size +=1
                cube_size = cube_size-1
                cube_height = cube_size
                while True:
                    kernel=np.ones((cube_height,cube_size))
                    eroded = ndimage.binary_erosion(target_hole,structure = kernel).astype(target_hole.dtype)
                    if eroded.sum() == 0:
                        break
                    else:
                        cube_height +=1
                cube_height = cube_height-1
                cube_width = cube_size
                while True:
                    kernel=np.ones((cube_height,cube_width))
                    eroded = ndimage.binary_erosion(target_hole,structure = kernel).astype(target_hole.dtype)
                    if eroded.sum() == 0:
                        break
                    else:
                        cube_width +=1
                cube_width = cube_width-1

                kernel=np.ones((cube_height,cube_width))
                eroded = ndimage.binary_erosion(target_hole,structure = kernel).astype(target_hole.dtype)

                cube = Brick(cube_width,cube_height)
                cubes.append(cube)
                cubes_xs.append(np.min(np.argwhere(eroded!=0),axis = 0)[1])
                cubes_ys.append(np.min(np.argwhere(eroded!=0),axis = 0)[0])
        ###########################################################################
        elif add_element_type == 'flexible':
            #using image body
            for target_label in target_labels:
                target_hole = np.where(label_image==target_label,1,0)
                #smooth edge
                nb_erosion = 0
                kernel_size = 3
                kernel=np.ones((kernel_size,kernel_size))
                eroded = ndimage.binary_erosion(target_hole,structure = kernel).astype(target_hole.dtype)
                label_image = label(eroded,background = 0,connectivity = 1)
                while label_image.max()>1:
                    eroded = ndimage.binary_erosion(eroded,structure = kernel).astype(target_hole.dtype)
                    label_image = label(eroded,background = 0,connectivity = 1)
                    nb_erosion+=1
                if label_image.max()==0:
                    continue
                dilated = ndimage.binary_dilation(eroded,structure = kernel).astype(target_hole.dtype)
                for i in range(nb_erosion): 
                    dilated = ndimage.binary_dilation(dilated,structure = kernel).astype(target_hole.dtype)
                stone = Stone()
                sucess,coord = stone.from_matrix(dilated)
                if not sucess:
                    continue
                if stone.width<3 or stone.height<3:
                    continue
                stone.id = get_cut_id()
                cubes.append(stone)
                cubes_xs.append(coord[0])
                cubes_ys.append(coord[1])

        return cubes,cubes_xs,cubes_ys
        

def build_wall_vendor(base, vendor, nb_stones_per_ite, allow_cutting = False,refill=False, vendor_type='similar',variant_sample = 1,construction_order='from_left', defined_sequence = []):
    plot_process = get_record_detail()['RECORD_PLACEMENT_IMG']
    _pre_label = None
    if get_record_time():
        with open(get_dir()+"time.txt", "w+") as f:
            f.write("iteration;number_placement;placement;typology evaluation;stabilization;kinematics evaluation\n")
    _sample_times = 0
    already_cut = False
    #_max_length = min(vendor.get_max_width(),base.matrix.shape[1]*0.5)
    _max_length = base.matrix.shape[1]
    _max_area = vendor.get_max_stone_size()
    if defined_sequence!=[]:
        for defined_id in defined_sequence:
            # find the stone index
            current_bricks = vendor.get_all_stones()
            for stone_index, stone in enumerate(current_bricks):
                if stone.id == defined_id[0]:
                    stone_90  = stone.rotated_by_90()
                    stone_180 = stone_90.rotated_by_90()
                    stone_270 = stone_180.rotated_by_90()                  
                        
                    if defined_id[3]==0:
                        current_bricks[stone_index] = stone
                        if defined_id[1]==0 and defined_id[2]==0:
                            trans_x, trans_y,_,_,_ = place_one_stone(stone_index,current_bricks,current_bricks,base)
                        else:
                            trans_x = defined_id[1]
                            trans_y = defined_id[2]
                        base.add_stone(stone, [trans_x, trans_y])
                    elif defined_id[3]==90:
                        current_bricks[stone_index] = stone_90
                        if defined_id[1]==0 and defined_id[2]==0:
                            trans_x, trans_y,_,_,_ = place_one_stone(stone_index,current_bricks,current_bricks,base)
                        else:
                            trans_x = defined_id[1]
                            trans_y = defined_id[2]
                        base.add_stone(stone_90, [trans_x, trans_y])
                    elif defined_id[3]==180:
                        current_bricks[stone_index] = stone_180
                        if defined_id[1]==0 and defined_id[2]==0:
                            trans_x, trans_y,_,_,_ = place_one_stone(stone_index,current_bricks,current_bricks,base)
                        else:
                            trans_x = defined_id[1]
                            trans_y = defined_id[2]
                        base.add_stone(stone_180, [trans_x, trans_y])
                    elif defined_id[3]==270:
                        current_bricks[stone_index] = stone_270
                        if defined_id[1]==0 and defined_id[2]==0:
                            trans_x, trans_y,_,_,_ = place_one_stone(stone_index,current_bricks,current_bricks,base)
                        else:
                            trans_x = defined_id[1]
                            trans_y = defined_id[2]
                        base.add_stone(stone_270, [trans_x, trans_y])
                    _ = current_bricks.pop(stone_index)
                    vendor.return_stones(current_bricks)
                    #loggind information
                    get_main_logger().critical(f"Place stone {stone.id} at {trans_x},{trans_y}")
                    break
    sample_trial = 0
    max_sample_trial = 5
    while True:
        #_______________________________________________________________________________________SAMPLE
        if vendor_type == 'random':
            current_bricks = vendor.get_random_stones(nb_stones_per_ite)
        elif vendor_type == 'similar':
            current_bricks = vendor.get_similar_stones(nb_stones_per_ite)
        elif vendor_type == 'variant':
            current_bricks = vendor.get_variant_stones(nb_stones_per_ite,variant_sample = variant_sample)
            #get all bricks if the id is in the current bricks list
            selected_ids = []
            for brick in current_bricks:
                selected_ids.append(brick.id)
            added_bricks = vendor.remove_stone_by_id(selected_ids)
            current_bricks.extend(added_bricks)
                    
        elif  vendor_type == 'same':
            current_bricks = vendor.get_same_stones(nb_stones_per_ite, pre_label = _pre_label)
        elif  vendor_type == 'full':
            current_bricks = vendor.get_all_stones()
        elif  vendor_type == 'full_replace':
            current_bricks = vendor.get_all_stones_replacing()
        elif vendor_type == 'sample_twice':
            if _sample_times%2==0:#sample to get  cluster number
                current_bricks = vendor.get_variant_stones(nb_stones_per_ite,variant_sample = variant_sample)
            else:
                current_bricks = vendor.get_same_stones(nb_stones_per_ite, pre_label = _pre_label, compensate=False,all_same_cluster=True)
            _sample_times+=1
            print(f"\nIteration {len(base.placed_bricks)}, Sample Time {_sample_times}\n")
            for b in current_bricks:
                print(f"Brick {b.id}, cluster {b.cluster}\n")
        #current_bricks = vendor.get_stones_all_labels(nb_stones_per_ite)
        if not vendor_type == 'sample_twice':
            if len(current_bricks) < nb_stones_per_ite:
                added_bricks = vendor.get_random_stones(
                    nb_stones_per_ite-len(current_bricks))
                current_bricks.extend(added_bricks)
        elif len(current_bricks)==1:
            print(f"Only one cluster in current data set")
            added_bricks = vendor.get_same_stones(nb_stones_per_ite, pre_label = current_bricks[0].cluster, compensate=False,all_same_cluster=True)
            current_bricks.extend(added_bricks)

        
        #_______________________________________________________________________________________PLACEMENT
        construction_step_i = len(base.placed_bricks)
        _attempt_events = {}  # trace: side -> filter event id of this sampling attempt
        if vendor_type == 'sample_twice':
            _phase = 'probe' if _sample_times % 2 == 1 else 'place'
        else:
            _phase = 'single'
        def construct_one_step(base, current_bricks, from_right=False):
            if from_right:
                base = base.create_mirrored_base()
                for i,brick_mirror in enumerate(current_bricks):
                    current_bricks[i] = brick_mirror.create_mirrored_stone(align_to_right=True)
            rotated_bricks_270 = []
            rotated_bricks_180 = []
            rotated_bricks_90 = []
            if get_rotate_state():
                for i, b in enumerate(current_bricks):
                    rotate_90 = b.rotated_by_90()
                    rotate_90.rotate_from_ori = -90
                    rotated_bricks_90.append(rotate_90)
                    rotate_180 = rotate_90.rotated_by_90()
                    rotate_180.rotate_from_ori = -180
                    rotated_bricks_180.append(rotate_180)
                    rotate_270 = rotate_180.rotated_by_90()
                    rotate_270.rotate_from_ori = -270
                    rotated_bricks_270.append(rotate_270)   
            brick = current_bricks[0]
            min_distance = float("+inf")
            # loop to choose one placement
            min_x = 0
            min_y = 0
            min_index = 0

            # occupied = np.argwhere(np.where(
            #     base.matrix != get_ignore_pixel_value(), base.matrix, 0))  # !return index
            # if occupied.any():
            #     # find bounding box
            #     _max_stone_bb = occupied.max(axis=0)
            #     _min_stone_bb = occupied.min(axis=0)
            #     bbox = np.zeros(base.matrix.shape)
            #     bbox[_min_stone_bb[0]:_max_stone_bb[0]+2,
            #         _min_stone_bb[1]:_max_stone_bb[1]+2] = 1

            #     # oriented contour
            #     img = base.matrix.astype(np.uint8)
            #     img_reverted = img*-1
            #     kernel_dilation_y = np.ones((get_dilation(), 1), np.uint8)
            #     kernel_erosion_y = np.ones((1, get_erosion()), np.uint8)
            #     kernel_dilation_x = np.ones((1, get_dilation()), np.uint8)
            #     kernel_erosion_x = np.ones((get_erosion(), 1), np.uint8)
            #     # horizontal contours:detect-expand-erode noise
            #     sobely = cv2.Sobel(img_reverted, cv2.CV_8U, 0, 1, ksize=-1)
            #     img_dilation_y = cv2.dilate(
            #         sobely, kernel_dilation_y, iterations=1)
            #     img_erosion_y = cv2.erode(
            #         img_dilation_y, kernel_erosion_y, iterations=1)
            #     # vertical contour:detect-expand-erode noise
            #     sobelx = cv2.Sobel(img_reverted, cv2.CV_8U, 1, 0, ksize=-1)
            #     img_dilation_x = cv2.dilate(
            #         sobelx, kernel_dilation_x, iterations=1)
            #     img_erosion_x = cv2.erode(
            #         img_dilation_x, kernel_erosion_x, iterations=1)

            #     contour = np.where(((sobely != 0)
            #                         | (sobelx != 0)), 1, 0)
            #     ocontour = np.where(((img_erosion_x != 0)
            #                         | (img_erosion_y != 0)), 1, 0)

            #     # void
            #     void = np.where(base.matrix == 0, 1, 0)

            #     # intersection of the above
            #     region = np.multiply(bbox, ocontour)
            #     region = np.multiply(region, void)
            #     if plot_process == True:
            #         save_matrix_plot(
            #             np.where(bbox != get_ignore_pixel_value(), bbox, 0), f"iteration_{construction_step_i}_bbox.png")
            #         save_matrix_plot(np.where(ocontour != get_ignore_pixel_value(
            #         ), ocontour, 0), f"iteration_{construction_step_i}_ocontour.png")
            #         save_matrix_plot(np.where(contour != get_ignore_pixel_value(
            #         ), contour, 0), f"iteration_{construction_step_i}_contour.png")
            #         save_matrix_plot(
            #             np.where(void != get_ignore_pixel_value(), void, 0), f"iteration_{construction_step_i}_void.png")
            #         save_matrix_plot(np.where(
            #             region != get_ignore_pixel_value(), region, 0), f"iteration_{construction_step_i}_region.png")

            #     locations = np.argwhere(region)

            # else:  # no stone placed
            #     locations = np.asarray([[0, 0]])
            #     region = np.zeros(base.matrix.shape)
            #     region[0, 0] = 1
            locations = np.asarray([[0, 0]])
            region = np.zeros(base.matrix.shape)
            region[0, 0] = 1

            # iterate over bricks:
            all_bricks = deepcopy(
                current_bricks+rotated_bricks_90+rotated_bricks_180+rotated_bricks_270)
            nb_stones = len(current_bricks)
            nb_poses = len(all_bricks)

            # container to store results for differnet stones
            min_x_list = []
            min_y_list = []
            min_distance_list = []
            filters_matrix = np.zeros((nb_poses, 6))
            times_matrix = np.zeros((nb_poses, 6))
            kine_list = []
            
            if not sequential_run:
                _nb_processor = get_number_cpu()
                _nb_processor = min(_nb_processor, nb_poses)
                for brick_index in range(0, nb_poses, _nb_processor):
                    inputs = []
                    for j in range(brick_index, min(brick_index+_nb_processor, nb_poses)):
                        input = (j, current_bricks, all_bricks, base,
                                locations, False,True,region)
                        inputs.append(input)
                    with Pool(_nb_processor) as p:
                        x_y_dist = p.starmap(place_one_stone, inputs)
                    for j in range(0, min(_nb_processor, nb_poses-brick_index)):
                        min_x_list.append(x_y_dist[j][0])
                        min_y_list.append(x_y_dist[j][1])
                        min_distance_list.append(x_y_dist[j][2])
                        filters_matrix[brick_index+j, :] = x_y_dist[j][3]
                        kine_list.append(x_y_dist[j][4])
                        times_matrix[brick_index+j, :] = x_y_dist[j][5]
            else:
                for brick_index in range(nb_poses):
                    min_x, min_y, min_distance,filters, kine,times = place_one_stone(
                        brick_index, current_bricks, all_bricks, base, locations, False,True,region)
                    min_x_list.append(min_x)
                    min_y_list.append(min_y)
                    min_distance_list.append(min_distance)
                    filters_matrix[brick_index, :] = filters
                    kine_list.append(kine)
                    times_matrix[brick_index, :] = times

            #write matrix to txt file
            if get_record_time():
                with open(get_dir()+"time.txt", "a+") as f:
                    np.savetxt(f, times_matrix,delimiter=';', newline='\n')
            # import copy
            # for i in range(nb_poses):
            #     base_temp = copy.deepcopy(base)
            #     base_temp.add_stone(all_bricks[i],[min_x_list[i],min_y_list[i]])
            #     base_temp.draw_last_action(file_name = f'iteration_{len(base_temp.placed_bricks)}_{all_bricks[i].id}_{all_bricks[i].width}.pdf')
            # # find the minimum distance
            # min_distance_list = np.array(min_distance_list)
            # min_index = np.argmin(min_distance_list)
            # min_x = min_x_list[min_index]
            # min_y = min_y_list[min_index]
            # min_distance = min_distance_list[min_index]

            # hierarchical filter, see selection.py. The default policy 'code_original' is the
            # author's cascade with filter_order = [1,2,3,4,0] ("brick wall width 24*12").
            #0 course_height, 1 global_density, 2 interlocking, 3 kinematic, 4 local_void, 5 local_void_center
            filters_matrix_ori = filters_matrix.copy()
            selection_weights = get_selection_weights()
            weights = [selection_weights[name] for name in FEATURE_NAMES]
            selection = select(filters_matrix_ori, weights, **get_filter_policy())
            if get_trace() is not None:
                _side = 'right' if from_right else 'left'
                _attempt_events[_side] = trace_event({'step': construction_step_i, 'side': _side,
                             'phase': _phase, 'sample_trial': sample_trial, 'vendor_type': vendor_type,
                             'construction_order': construction_order, 'nb_stones': nb_stones},
                            base, vendor, current_bricks, all_bricks, min_x_list, min_y_list,
                            min_distance_list, filters_matrix_ori, weights, selection)
            filters_matrix = selection.weighted
            min_index = selection.index
            with open(get_dir()+"filter.txt", "a+") as f:
                f.write(filter_log(construction_step_i, [b.id for b in all_bricks], selection))
            if min_index is None:
                get_main_logger().warning(
                    f"No possible placement for any of the {nb_stones} given")
                return False,None,None,None,None,-np.inf,-1*np.ones((filters_matrix.shape[1]))

            # else:
            #     ## height, density, interlocking, kine, local_void,local_void_center
            #     filters_matrix_ori = filters_matrix.copy()
            #     result_0 = np.argwhere(filters_matrix[:, 3] >0).reshape((1, -1))
            #     if result_0[0].shape[0]>0:
            #         weight_matrix = np.zeros((6,6))#diagnal matrix
            #         selection_weights = get_selection_weights()
            #         weight_matrix[0,0] = selection_weights['course_height']
            #         weight_matrix[1,1] = selection_weights['global_density']
            #         weight_matrix[2,2] = selection_weights['interlocking']
            #         weight_matrix[3,3] = selection_weights['kinematic']
            #         weight_matrix[4,4] = selection_weights['local_void']
            #         weight_matrix[5,5] = selection_weights['local_void_center']
            #         filters_matrix = np.matmul(filters_matrix, weight_matrix)
            #         kine_ok_filter_matrix = filters_matrix[filters_matrix[:, 3]>=0]
            #         filter3 = np.argwhere(filters_matrix[:, 3] >= np.average(
            #             kine_ok_filter_matrix[:, 3])).reshape((1, -1))
            #         filter0 = np.argwhere(filters_matrix[:, 0] >= np.average(
            #             filters_matrix[:, 0])).reshape((1, -1))
            #         result_1 = np.intersect1d(filter3, filter0)
            #         result_1 = np.intersect1d(result_1, result_0)
            #         if result_1.shape[0] > 0:
            #             filter2 = np.argwhere(filters_matrix[:, 2] >= np.average(
            #                 filters_matrix[:, 2])).reshape((1, -1))
            #             result_2 = np.intersect1d(result_1, filter2)
            #             if result_2.shape[0] > 0:
            #                 filter1 = np.argwhere(filters_matrix[:, 1] >= np.average(
            #                     filters_matrix[:, 1])).reshape((1, -1))
            #                 result_3 = np.intersect1d(result_2, filter1)
            #                 if result_3.shape[0] > 0:
            #                     filter4 = np.argwhere(filters_matrix[:, 5] >= np.average(
            #                         filters_matrix[:, 5])).reshape((1, -1))
            #                     result_4 = np.intersect1d(result_3, filter4)
            #                     if result_4.shape[0] > 0:
            #                         filter5 = np.argwhere(filters_matrix[:, 4] >= np.average(
            #                             filters_matrix[:, 4])).reshape((1, -1))
            #                         result_5 = np.intersect1d(result_4, filter5)
            #                         if result_5.shape[0] > 0:
            #                             min_index = result_5[np.argmax(
            #                                     filters_matrix[result_5, 3]+filters_matrix[result_5,0]+filters_matrix[result_5,2]+filters_matrix[result_5,1]+filters_matrix[result_5,4]+filters_matrix[result_5,5])]
            #                         else:
            #                             min_index = result_4[np.argmax(
            #                             filters_matrix[result_4, 3]+filters_matrix[result_4,0]+filters_matrix[result_4,2]+filters_matrix[result_4,1]+filters_matrix[result_4,5])]
            #                     else:
            #                         min_index = result_3[np.argmax(
            #                             filters_matrix[result_3, 3]+filters_matrix[result_3,0]+filters_matrix[result_3,2]+filters_matrix[result_3,1])]
            #                 else:
            #                     min_index = result_2[np.argmax(
            #                         filters_matrix[result_2, 3]+filters_matrix[result_2,0]+filters_matrix[result_2,2])]
            #             else:
            #                 min_index = result_1[np.argmax(filters_matrix[result_1, 3]+filters_matrix[result_1, 0])]
            #         else:
            #             min_index = np.argmax(filters_matrix[:, 3])
            #     else:
            #         get_main_logger().warning(
            #             f"No possible placement for any of the {nb_stones} given")
            #         return False,None,None,None,None,-np.inf

            min_x = min_x_list[min_index]
            min_y = min_y_list[min_index]
            min_distance = min_distance_list[min_index]

            if min_distance == float("+inf"):
                get_main_logger().warning(
                    f"No possible placement for any of the {nb_stones} given")
                return False,None,None,None,None,-np.inf,-1*np.ones((filters_matrix.shape[1]))
            if filters_matrix_ori[min_index,3] <= 0:
                get_main_logger().warning(
                    f"No possible placement for any of the {nb_stones} given")
                return False,None,None,None,None,-np.inf,-1*np.ones((filters_matrix.shape[1]))

            brick = all_bricks[min_index]
            get_main_logger().critical(
                f"Stone {brick.id} is selected for the {construction_step_i}th placement, with limit force {kine_list[min_index]}")
            # ! make sure that the brick is the one from solution
            #get_main_logger().debug(f"{min_distance_list[min_index]}")
            #get_main_logger().debug(min_distance_list)
            if from_right:
                brick = brick.create_mirrored_stone(align_to_right = True)
                print("WORLD SIZE",brick._w)
                print("Solved x",min_x)
                print("Brick size, ", brick.width)
                min_x = brick._w - min_x-brick.width
            return True,brick, min_index, min_x, min_y,np.sum(filters_matrix[min_index, :]),filters_matrix[min_index, :]

        _test_base = deepcopy(base)
        _test_bricks = deepcopy(current_bricks)
        if construction_order == "two_sides":
            continue_construction_left,brick_left, min_index_left, min_x_left, min_y_left,metric_sum_left,features_left = construct_one_step(_test_base, _test_bricks,from_right=False)
            continue_construction_right,brick_right, min_index_right, min_x_right, min_y_right,metric_sum_right,features_right = construct_one_step(_test_base, _test_bricks,from_right=True)
            
            if features_right[1] > features_left[1]:
                brick = brick_right
                min_index = min_index_right
                min_x = min_x_right
                min_y = min_y_right
                continue_construction = continue_construction_right
                features_best = features_right
            else:
                brick = brick_left
                min_index = min_index_left
                min_x = min_x_left
                min_y = min_y_left
                continue_construction = continue_construction_left
                features_best = features_left
        elif construction_order == "from_right":
            continue_construction,brick, min_index, min_x, min_y,_,features_best = construct_one_step(_test_base, _test_bricks,from_right=True)
        elif construction_order == "from_left":
            continue_construction,brick, min_index, min_x, min_y,_ ,features_best= construct_one_step(_test_base, _test_bricks,from_right=False)

        if get_trace() is not None:
            # side choice above: right only if its weighted GDS is strictly greater (ties -> left)
            if construction_order == "two_sides":
                _sides = {'left': [continue_construction_left, min_index_left, features_left[1]],
                          'right': [continue_construction_right, min_index_right, features_right[1]]}
                _chosen = 'right' if features_right[1] > features_left[1] else 'left'
            else:
                _chosen = 'right' if construction_order == "from_right" else 'left'
                _sides = {_chosen: [continue_construction, min_index, features_best[1]]}
            _attempt = {'step': construction_step_i, 'phase': _phase, 'sample_trial': sample_trial,
                        'vendor_type': vendor_type, 'construction_order': construction_order,
                        'sampled_ids': [b.id for b in current_bricks],
                        'sides': {s: {'event': _attempt_events.get(s), 'ok': v[0], 'index': v[1], 'gds_weighted': v[2]}
                                  for s, v in _sides.items()},
                        'chosen_side': _chosen,
                        'side_tie': construction_order == "two_sides" and features_right[1] == features_left[1]}
            if continue_construction:
                _label = min_index // len(current_bricks) * 90
                _attempt.update(stone_id=brick.id, cluster=brick.cluster, rotation_label=_label,
                                rotation_world=_label if _chosen == 'left' else (360 - _label) % 360,
                                x_world=min_x, y=min_y)

        if not continue_construction:
            if get_trace() is not None:
                trace_commit({**_attempt, 'outcome': 'no_placement',
                              'termination': None if sample_trial < max_sample_trial else 'sample_trials_exhausted'})
            if sample_trial<max_sample_trial:
                #break
                sample_trial+=1
                if vendor_type == 'sample_twice':
                    _sample_times-=1
                vendor.return_stones(current_bricks)
                print(f"Trial number {sample_trial}")
                continue
            else:
                break

        else:
            sample_trial = 0
        _pre_nb_stones = len(base.placed_bricks)
        if vendor_type == "sample_twice" and _sample_times%2==1:
            _pre_label = brick.cluster
            vendor.return_stones(current_bricks)
            if get_trace() is not None:
                trace_commit({**_attempt, 'outcome': 'probe', 'probe_cluster': _pre_label, 'termination': None})
        else:
            place_the_solution = True

            if allow_cutting == True and features_best[1]<0.5 and already_cut == False:
                # if yes, we fill the layer instead of puting the current stone
                already_cut = True
                cubes,cube_xs,cube_ys = get_cut_stone(base, max_area =_max_area,min_area = _max_area/10,max_width=_max_length)
                if cubes:
                    for cube_index in range(len(cubes)):
                        cube = cubes[cube_index]
                        future_base = deepcopy(base)
                        future_base.add_stone(cube, [cube_xs[cube_index], cube_ys[cube_index]])
                        kine = evaluate_kine(future_base,load = 'tilting_table')
                        if kine>0:
                            base.add_stone(cube, [cube_xs[cube_index], cube_ys[cube_index]])
                            get_main_logger().critical(f"Cutting stone {cube.id} at {cube_xs[cube_index], cube_ys[cube_index]}")
                        else:
                            get_main_logger().critical("Try to cut stone, but not stable")
                            continue
                    _sample_times-=2
                    vendor.return_stones(current_bricks)
                    place_the_solution = False
                    if get_trace() is not None:
                        trace_commit({**_attempt, 'outcome': 'cut', 'termination': None})
                else:
                    #place the solved stone
                    place_the_solution = True
            if place_the_solution:
                already_cut = False
                

                adding_ok = base.add_stone(brick, [min_x, min_y])
                _pre_label = brick.cluster
                _new_nb_stones = len(base.placed_bricks)
                if get_trace() is not None:
                    # the placed rock can differ from the requested pose (overlap clipping, stabilization)
                    _rock = base.placed_rocks[-1] if base.placed_rocks else None
                    _placed = {'add_stone_ok': adding_ok, 'bricks_added': _new_nb_stones - _pre_nb_stones}
                    if adding_ok and _rock is not None and _rock.id == brick.id:
                        _rows, _cols = np.nonzero(_rock.matrix)
                        _placed.update(placed_bbox=[_cols.min(), _rows.min(), _cols.max(), _rows.max()],
                                       placed_rot_angle=_rock.rot_angle, placed_rotate_from_ori=_rock.rotate_from_ori)
                if _new_nb_stones-_pre_nb_stones > 1:
                    get_main_logger().critical(
                        f'Add wedge to stone {brick.id}')
                if plot_process == True and adding_ok == True:
                    save_matrix(
                        base.placed_bricks[construction_step_i].matrix, f"iteration_{construction_step_i}_stone.png")
                base.draw_assembly(
                    save_draw=True, file_name=f'AA_iteration_{construction_step_i}_cluster.pdf',sequence = False,number_type ='cluster')
                base.draw_assembly(
                    save_draw=True, file_name=f'AA_iteration_{construction_step_i}.pdf',sequence = True,number_type ='id')

                # save base object to file by pickle
                if get_dump():
                    with open(get_dir()+"base.pkl", "wb") as filehandler:
                        pickle.dump(base, filehandler)
                # print(f"brick added to base: {brick.width} * {brick.height}")
                # print(f"brick removed: {min_brick.width} * {min_brick.height}")
                with open(get_dir()+"decisions.txt", "a+") as f:
                    if min_index < len(current_bricks):
                        f.write(f"{construction_step_i};{brick.id};{min_x};{min_y};0\n")
                        _ = current_bricks.pop(min_index)
                        # _ = rotated_bricks_90.pop(min_index)
                        # _ = rotated_bricks_180.pop(min_index)
                        # _ = rotated_bricks_270.pop(min_index)
                    elif min_index < 2*len(current_bricks):
                        f.write(f"{construction_step_i};{brick.id};{min_x};{min_y};90\n")
                        _nb_bricks = len(current_bricks)
                        _ = current_bricks.pop(min_index-2*_nb_bricks)
                        # _ = rotated_bricks_90.pop(min_index-2*_nb_bricks)
                        # _ = rotated_bricks_180.pop(min_index-2*_nb_bricks)
                        # _ = rotated_bricks_270.pop(min_index-2*_nb_bricks)

                    elif min_index < 3*len(current_bricks):
                        f.write(f"{construction_step_i};{brick.id};{min_x};{min_y};180\n")
                        _nb_bricks = len(current_bricks)
                        _ = current_bricks.pop(min_index-3*_nb_bricks)
                        # _ = rotated_bricks_90.pop(min_index-3*_nb_bricks)
                        # _ = rotated_bricks_180.pop(min_index-3*_nb_bricks)
                        # _ = rotated_bricks_270.pop(min_index-3*_nb_bricks)
                    else:
                        f.write(f"{construction_step_i};{brick.id};{min_x};{min_y};270\n")
                        _nb_bricks = len(current_bricks)
                        _ = current_bricks.pop(min_index-4*_nb_bricks)
                        # _ = rotated_bricks_90.pop(min_index-4*_nb_bricks)
                        # _ = rotated_bricks_180.pop(min_index-4*_nb_bricks)
                        # _ = rotated_bricks_270.pop(min_index-4*_nb_bricks)
                    f.close()

                if not vendor_type == 'full_replace':
                    #print("Returning stones ", len(current_bricks))
                    #iterate all bricks, eliminate bricsk of the same id as the chosen brick
                    pop_index = []
                    ref_ids = []
                    for angle in range(-80,90,10):
                        ref_ids.append(base.placed_rocks[-1].id+angle*100)
                    for brick_index, brick in enumerate(current_bricks):
                        if brick.id in ref_ids:
                            pop_index.append(brick_index)
                    for index in sorted(pop_index, reverse=True):
                        del current_bricks[index]
                    vendor.return_stones(current_bricks)
                    get_main_logger().critical(f"{len(vendor.stones.keys())} stones remaining in the set")
                #check termination
                if get_trace() is not None:
                    trace_commit({**_attempt, **_placed, 'outcome': 'placed',
                                  'termination': 'inventory_empty' if vendor.get_number_stones() == 0 else
                                  'filled' if base.get_filling()/(base._w*base._h) > 0.999 else None})
                if vendor.get_number_stones() == 0:
                    break
                if base.get_filling()/(base._w*base._h) > 0.999:
                    break
    # fill the layer instead of puting the current stone
    if allow_cutting == True:
        cubes,cube_xs,cube_ys = get_cut_stone(base, max_area =_max_area,min_area = _max_area/10,max_width=_max_length)
        if cubes:
            for cube_index in range(len(cubes)):
                cube = cubes[cube_index]
                future_base = deepcopy(base)
                future_base.add_stone(cube, [cube_xs[cube_index], cube_ys[cube_index]])
                kine = evaluate_kine(future_base,load = 'tilting_table')
                if kine>0:
                    base.add_stone(cube, [cube_xs[cube_index], cube_ys[cube_index]])
                    get_main_logger().critical(f"Cutting stone {cube.id} at {cube_xs[cube_index], cube_ys[cube_index]}")
                else:
                    get_main_logger().critical("Try to cut stone, but not stable")
                    continue
    if get_dump():
        with open(get_dir()+"base.pkl", "wb") as filehandler:
            pickle.dump(base, filehandler)
    return base