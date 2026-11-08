from scipy import ndimage
from ..utils.kine_2d import ContPoint, ContType
from ..utils.region_memo import stone_region
import numpy as np
from skimage.measure import regionprops
from ..utils.kine_2d import Element, update_elem_disp
import cv2
from ..utils.constant import get_ignore_pixel_value, get_world_size, get_mu, get_head_joint_state, get_kine_precision, get_dir
from ..utils.kine_2d import cal_Aglobal, solve_force_rigid, plot_elem_contp
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

tab20_cm = cm.get_cmap('tab20')
newcolors = np.concatenate([tab20_cm(np.linspace(0, 1, 20))] * 13, axis=0)
white = np.array([255/256, 255/256, 255/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)


# rot_cmap=cm.get_cmap('RdBu')
# rot_cmap[0] = white
class Parameter:

    # kinematics
    boundary = "tilting-table"
    stress = 30/100
    mu = 1.0
    fc = 0
    cohesion = 0
    density = 18
    calibrate_head_joint = False
    ignore_head_joint = False
    min_contp_distance = 1000
    offset = 1
    contact_detection_method = "uniform"
    ctname = "friction"


parameters = Parameter()
parameters_head = Parameter()
parameters_mortar = Parameter()

WALL_CONTACTS_KEPT = 4  # wall images remembered per process
WALL_CONTACTS = {}


def plot_failure_mec(base_image, base_image_id, elems, LA, direction='right',plot_disp = 'euclidean',factor = 1e4):
    width = int(base_image.shape[1] * 1)
    height = int(base_image.shape[0] * 1)
    dim = (width, height)
    factor = factor
    moved_image = np.zeros(base_image.shape)
    # moved_image = cv2.resize(
    #     moved_image, dim, interpolation=cv2.INTER_NEAREST)
    max_rot = 0 
    max_x = 0
    max_y = 0
    max_euc = 0
    for key, element in elems.items():
        if abs(element.displacement[2]) > max_rot:
            max_rot = abs(element.displacement[2])
        if abs(element.displacement[0]) > max_x:
            max_x = abs(element.displacement[0])
        if abs(element.displacement[1]) > max_y:
            max_y = abs(element.displacement[1])
        if np.sqrt(abs(element.displacement[0])**2+abs(element.displacement[1])**2) > max_euc:
            max_euc = np.sqrt(abs(element.displacement[0])**2+abs(element.displacement[1])**2)
    for key, element in elems.items():
        if plot_disp == 'rotation':
            color_index = 0.01+(abs(element.displacement[2])/max_rot)
        elif plot_disp == 'x':
            color_index = 0.01+(abs(element.displacement[0])/max_x)
        elif plot_disp == 'y':
            color_index = 0.01+(abs(element.displacement[1])/max_y)
        elif plot_disp == 'euclidean':
            color_index = 0.01+(np.sqrt((abs(element.displacement[0])**2+abs(element.displacement[1])**2))/max_euc)
        else:
            color_index = 0.01
        #stone_pixels = np.where(base_image == element.id, base_image_id+1, 0)
        stone_pixels = np.where(base_image == element.id, color_index, 0)
        rot_center = element.center
        
        rot_mat = cv2.getRotationMatrix2D(
            (rot_center[0], rot_center[1]), element.displacement[2]*180/np.pi*factor, 1.0)
        rotated_stone_pixels = cv2.warpAffine(
            stone_pixels, rot_mat, stone_pixels.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
        T = np.float32([[1, 0, element.displacement[0]*factor],
                        [0, 1, element.displacement[1]*factor]])
        translated_stone_pixels = cv2.warpAffine(
            rotated_stone_pixels, T, base_image.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
        # translated_stone_pixels = cv2.resize(
        #     translated_stone_pixels, dim, interpolation=cv2.INTER_NEAREST)

        moved_image += translated_stone_pixels

    plt.clf()
    fig, ax = plt.subplots()
    #moved_image = np.where(moved_image == 1, 10, moved_image)
    moved_image = np.where(moved_image == 0, 0.5, moved_image)
    #moved_image[moved_image==0.5]=np.nan
    _plot = ax.imshow(moved_image, cmap='seismic',interpolation='none', vmin=0,vmax=1)
    # for key, value in elems.items():
    #     plt.scatter(value.center[0]*scale_+value.displacement[0]*factor*scale_,
    #                 value.center[1]*scale_+value.displacement[1]*factor*scale_, marker='^', c=color[value.id])

    # plt.axis('off')
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # text_kwargs = dict(ha='center', va='center', fontsize=20, color='C1')
    # for i, stone_i in enumerate(self.placed_rocks):
    #     plt.text(self.rock_centers[i][0],
    #             self.rock_centers[i][1], str(stone_i.id), **text_kwargs)

    # ax.text(0.5, 0.9, f"Limit load: {LA:.02f}\nForce direction {direction}", horizontalalignment='center',
    #         verticalalignment='center', transform=ax.transAxes, fontsize=16, color='black')
    plt.gca().invert_yaxis()
    plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(_plot,cax = cax,orientation="horizontal")
    if plot_disp == 'rotation':
        cbar.set_label('element rotation/max rotation')
    elif plot_disp == 'x':
        cbar.set_label('element x displacement/max x displacement')
    elif plot_disp == 'y':
        cbar.set_label('element y displacement/max y displacement')
    elif plot_disp == 'euclidean':
        cbar.set_label('element displacement/max displacement')
    with open(get_dir()+"mechanism.csv", "a+") as f:  # the limit load, which only the image's name held before
        f.write(("side,limit_load\n" if f.tell() == 0 else "") + "{},{}\n".format(direction, LA))
    plt.savefig(get_dir()+"img/"+direction+str(LA)+'_mechanism.png', dpi=2400, transparent=True)


def _contact_image(matrix):
    """The wall image the contacts are found in: pixel ids as uint8, without the pixels to ignore."""
    img = matrix.astype(np.uint8)
    return np.where(img != get_ignore_pixel_value(), img, 0)


def evaluate_kine(final_base, save_failure=False, load='linear_acceleration', wall=None):
    """Evaluate the limit tilting angle of the placed stones

    :param final_base: The base to be evaluated
    :type final_base: Stonepacker2D.base.Base
    :return: The limit tilting angle of the placed stones, minimal of two direction
    :rtype: float
    """
    img = _contact_image(final_base.matrix)
    # print(img.shape)
    elems = base_to_elements(final_base)
    if wall is None:
        contps, Aglobal = contps_of_image(elems, img), None
    else:
        contps, Aglobal = contact_matrix(elems, img, _contact_image(wall))
    # ****************************************************************
    # ******** Tilting Table Load********************************
    # ****************************************************************
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [element.mass, 0, 0]

    # ****************************************************************
    # ******** Horizontal Load on Last Element ********************************
    # ****************************************************************
    # last_element_id = element_id
    # total_mass = 0
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [0, 0, 0]
    #     total_mass += element.mass
    # total_mass -= elems[last_element_id].mass
    # elems[last_element_id].dl = [0, -total_mass, 0]
    # elems[last_element_id].ll = [total_mass, 0, 0]

    # *****************************************************
    # ********** Linear Horizontal Load ***************************
    # *****************************************************
    _, wall_height = get_world_size()
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [
    #         (1-math.cos(math.pi*element.center[1]/(2*wall_height)))*element.mass, 0, 0]
    if load == 'linear_acceleration':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [
                (element.center[1]/wall_height)*element.mass, 0, 0]
    elif load == 'tilting_table':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [element.mass, 0, 0]

    if Aglobal is None:
        Aglobal = cal_Aglobal(elems, contps)
    #plot_elem_contp(elems, contps)
    one_direction_solution = solve_force_rigid(elems, contps, Aglobal)
    one_direction = one_direction_solution['limit_force']
    _plot_matrix = final_base.cluster_matrix.copy()
    if save_failure:
        update_elem_disp(elems, one_direction_solution['displacements'])
        plot_failure_mec(final_base.matrix, final_base.cluster_matrix,
                         elems, one_direction, direction='right',factor = 5e3)
        # plot_failure_mec(final_base.matrix, final_base.id_matrix,
        #                  elems, one_direction, direction='right')

    # ****************************************************************
    # ******** Tilting Table Load********************************
    # ***********************************************************
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [-element.mass, 0, 0]
    # ****************************************************************
    # ******** Horizontal Load on Last Element********************************
    # ****************************************************************
    # elems[last_element_id].ll = [-total_mass, 0, 0]
    # *****************************************************
    # ********** Linear Horizontal Load ***************************
    # *****************************************************
    # for element in elems.values():
    #     element.dl = [0, -element.mass, 0]
    #     element.ll = [-(1-math.cos(math.pi*element.center[1] /
    #                     (2*wall_height)))*element.mass, 0, 0]
    if load == 'linear_acceleration':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [-(element.center[1]/wall_height)*element.mass, 0, 0]
    elif load == 'tilting_table':
        for element in elems.values():
            element.dl = [0, -element.mass, 0]
            element.ll = [-element.mass, 0, 0]
    another_direction_solution = solve_force_rigid(elems, contps, Aglobal)
    another_direction = another_direction_solution['limit_force']
    if save_failure:
        update_elem_disp(elems, another_direction_solution['displacements'])
        # plot_failure_mec(final_base.matrix,
        #                  final_base.id_matrix, elems, another_direction, direction='left')
        plot_failure_mec(final_base.matrix, _plot_matrix,
                         elems, another_direction, direction='left',factor = 5e3)
        #print(one_direction, another_direction)
    # max_height = 0
    # for element in elems.values():
    #     if element.center[1] > max_height:
    #         max_height = element.center[1]
    #return 1
    return round(min(one_direction, another_direction), get_kine_precision())


def _append_contps(contps, maxPointID, img, index, coords, anta_rows, anta_cols, tangent, normal, conttype, ignore_zero):
    # one boundary direction, in the order of `index`; ContPoints are built last
    cand_ids = img[index[:, 0], index[:, 1]]
    anta_ids = img[anta_rows, anta_cols]
    keep = (cand_ids != 0) & (anta_ids != 0) if ignore_zero else np.ones(len(index), dtype=bool)
    for x, y, cand_id, anta_id in zip(coords[keep, 0], coords[keep, 1], cand_ids[keep], anta_ids[keep]):
        contps[maxPointID] = ContPoint(maxPointID, [x, y], cand_id, anta_id, tangent, normal, conttype)
        maxPointID += 1
    return maxPointID


def _boundary_directions(img):
    """The four kinds of boundary pixel of `img`: (pixels, contact coordinates, neighbour rows, neighbour columns, tangent,
    normal, whether it is a head joint), in the order the contact points are numbered."""
    # CONVOLVE: https://towardsdatascience.com/image-derivative-8a07a4118550
    directions = []

    # pixels whose value is different from pixel on the right
    normal = [-1, 0]
    tangent = [0, -1]
    Kx = np.array([[-1, 1, 0]])
    Fx_1 = ndimage.convolve(img, Kx)
    Kx = np.array([[1, -1, 0]])
    Fx_2 = ndimage.convolve(img, Kx)
    matrix_x_right = np.sum([Fx_1, Fx_2], axis=0)
    #matrix_x_right = np.multiply(img,matrix_x_right)
    index_x_right = np.argwhere(matrix_x_right > 0)
    coords = np.flip(np.sum([index_x_right, [[0, 0.5]]], axis=0), axis=1)
    directions.append((index_x_right, coords, index_x_right[:, 0], np.minimum(
        index_x_right[:, 1]+1, img.shape[1]), tangent, normal, True))

    # pixels whose value is different from pixel on the left
    normal = [1, 0]
    tangent = [0, 1]
    Kx = np.array([[0, -1, 1]])
    Fx_1 = ndimage.convolve(img, Kx)
    Kx = np.array([[0, 1, -1]])
    Fx_2 = ndimage.convolve(img, Kx)
    matrix_x_left = np.sum([Fx_1, Fx_2], axis=0)
    matrix_x_left = np.multiply(img, matrix_x_left)
    index_x_left = np.argwhere(matrix_x_left > 0)
    coords = np.flip(np.sum([index_x_left, [[0, -0.5]]], axis=0), axis=1)
    directions.append((index_x_left, coords, index_x_left[:, 0], np.maximum(
        index_x_left[:, 1]-1, 0), tangent, normal, True))

    # pixels whose value is different from pixel below
    normal = [0, -1]
    tangent = [1, 0]
    Ky = np.array([[-1], [1], [0]])
    Fy_1 = ndimage.convolve(img, Ky)
    Ky = np.array([[1], [-1], [0]])
    Fy_2 = ndimage.convolve(img, Ky)
    matrix_y_below = np.sum([Fy_1, Fy_2], axis=0)
    matrix_y_below = np.multiply(img, matrix_y_below)
    index_y_below = np.argwhere(matrix_y_below > 0)
    coords = np.flip(np.sum([index_y_below, [[+0.5, 0]]], axis=0), axis=1)
    directions.append((index_y_below, coords, np.minimum(
        index_y_below[:, 0]+1, img.shape[0]), index_y_below[:, 1], tangent, normal, False))

    # pixels whose value is different from pixel above
    normal = [0, 1]
    tangent = [-1, 0]
    Ky = np.array([[0], [1], [-1]])
    Fy_1 = ndimage.convolve(img, Ky)
    Ky = np.array([[0], [-1], [1]])
    Fy_2 = ndimage.convolve(img, Ky)
    matrix_y_above = np.sum([Fy_1, Fy_2], axis=0)
    matrix_y_above = np.multiply(img, matrix_y_above)
    index_y_above = np.argwhere(matrix_y_above > 0)
    coords = np.flip(np.sum([index_y_above, [[-0.5, 0]]], axis=0), axis=1)
    directions.append((index_y_above, coords, np.maximum(
        index_y_above[:, 0]-1, 0), index_y_above[:, 1], tangent, normal, False))
    return directions


def pixel_to_mesh(img,ignore_zero = True):
    contps = dict()
    maxPointID = 0
    parameters.mu = get_mu()

    _conttype = ContType("friction", parameters)
    parameters_head.mu = get_mu()
    if get_head_joint_state() == True:
        parameters_head.mu = 0
    _conttype_head = ContType("friction", parameters_head)

    for index, coords, anta_rows, anta_cols, tangent, normal, head in _boundary_directions(img):
        maxPointID = _append_contps(contps, maxPointID, img, index, coords, anta_rows, anta_cols, tangent, normal,
                                    _conttype_head if head else _conttype, ignore_zero)

    ####################################################################################################
    # print(len(contps))
    # print(len(contps))
    # for key, contp in contps.copy().items():
    #     n = np.asarray(contp.orientation[1])
    #     t = np.asarray(contp.orientation[0])
    #     for key_counter, contp_counter in contps.copy().items():
    #         n_counter = np.asarray(contp_counter.orientation[1])
    #         t_counter = np.asarray(contp_counter.orientation[0])
    #         # print(np.sum(np.multiply(n, n_counter)))
    #         # print(np.sum(np.multiply(n, n_counter)) == 0)
    #         # print(contp.coor == contp_counter.coor)
    #         # print(contp.cand == contp_counter.cand)
    #         # print(contp.anta == contp_counter.anta)
    #         if (contp.coor == contp_counter.coor):
    #             if np.sum(np.multiply(n, n_counter)) == 0:
    #                 if contp.cand == contp_counter.cand:
    #                     if contp.anta == contp_counter.anta:
    #                         # print("!!!!!!!!!!!!!!!!!")
    #                         if key in list(contps.keys()) and key_counter in list(contps.keys()):
    #                             del contps[key]
    #                             del contps[key_counter]
    #                             new_n = (n+n_counter) / \
    #                                 np.linalg.norm(n+n_counter)
    #                             _n_3d = np.asarray([new_n[0], new_n[1], 0])
    #                             _v = np.asarray([0, 0, 1])
    #                             _t = np.cross(_n_3d, _v)
    #                             new_t = _t[0:2]/np.linalg.norm(_t[0:2])
    #                             print(maxPointID, [
    #                                 contp.coor[0], contp.coor[1]], contp.cand, contp.anta, new_n.tolist(), new_t.tolist())

    #                             contps[maxPointID] = ContPoint(maxPointID, [
    #                                 contp.coor[0], contp.coor[1]], contp.cand, contp.anta, new_n.tolist(), new_t.tolist(), contp.cont_type)
    #                             # new_contps[(contp.coor[0], contp.coor[1], contp.cand, contp.anta)] = ContPoint(maxPointID, [
    #                             #     contp.coor[0], contp.coor[1]], contp.cand, contp.anta, ((n+n_counter)/1.414).to_list(), ((t+t_counter)/1.414).to_list(), contp.conttype)
    #                             # print(new_contps)
    #                             maxPointID += 1
    # print(len(contps))
    ####################################################################################################
    # for key, value in contps.items():
    #     if value.anta == 39:
    #         plt.imshow(np.flip(img, axis=0))
    #         plt.scatter(value.coor[0], value.coor[1])
    #         plt.show()
    return contps


def base_to_elements(final_base):
    elems = dict()
    for i, stone in enumerate(final_base.placed_bricks):
        element_id = i+1
        region = stone_region(stone)
        if not region:
            print(stone.matrix)
            print(stone.id)
            if np.argwhere(stone.matrix).shape[0]==0:
                continue
            else:
                min_index = np.min(np.argwhere(stone.matrix), axis=0)
                max_index = np.max(np.argwhere(stone.matrix), axis=0)
                center = [0.5*(min_index[1]+max_index[1]),
                        0.5*(min_index[0]+max_index[0])]
        else:
            center = [region[0][1], region[0][0]]
        #edges = cv2.Canny(stone.matrix.astype(np.uint8), 0, 0)
        #vertices = np.flip(np.argwhere(edges > 0), axis=1)
        vertices = [[center[0]-stone.width/2, center[1]-stone.height/2], [center[0]-stone.width/2, center[1]+stone.height/2],
                    [center[0]+stone.width/2, center[1]+stone.height/2], [center[0]+stone.width/2, center[1]-stone.height/2]]
        if i == 0:
            elems[element_id] = Element(
                element_id, center, region[1], vertices, type='ground')
        else:
            elems[element_id] = Element(
                element_id, center, region[1], vertices, type='stone')

    return elems


def _pair_geometry(normals, coordinates):
    """Both ends and the axes of the contact of one pair of elements; None when its normals cancel out."""
    averaged_normal = normals.mean(axis=0)
    norm = np.linalg.norm(averaged_normal)
    if norm == 0:
        return None
    # the cross product of (n0, n1, 0) and (0, 0, 1), term by term
    tangent = np.array([averaged_normal[1] - 0.0, 0.0 - averaged_normal[0]])
    new_t = tangent / np.linalg.norm(tangent)
    new_n = averaged_normal / norm
    # sort by projection to tangent direction
    projections = np.matmul(coordinates, np.asarray([[new_t[0]], [new_t[1]]]))
    coordinate_sort_indices = np.argsort(projections.reshape(1, -1))[0]
    return (
        coordinates[coordinate_sort_indices[0], :],
        coordinates[coordinate_sort_indices[-1], :],
        new_t.tolist(),
        new_n.tolist(),
    )


def _add_pair(cleaned_contps, maxPointID, value_cand, value_anta, geometry):
    """Add the two contact points of one pair of elements; returns the next point id."""
    if geometry is None:
        return maxPointID
    coordinate_end1, coordinate_end2, new_t, new_n = geometry
    if value_cand.type == 'mortar' and value_anta.type == 'mortar':
        _conttype = ContType("friction_fc_cohesion", parameters)
    else:
        _conttype = ContType("friction", parameters)
    cleaned_contps[maxPointID] = ContPoint(
        maxPointID, [coordinate_end1[0], coordinate_end1[1]], value_cand.id, value_anta.id, list(new_t), list(new_n),
        _conttype)
    maxPointID += 1
    cleaned_contps[maxPointID] = ContPoint(
        maxPointID, [coordinate_end2[0], coordinate_end2[1]], value_cand.id, value_anta.id, list(new_t), list(new_n),
        _conttype)
    return maxPointID + 1


def _cleaned_pair(cleaned_contps, maxPointID, value_cand, value_anta, normals, coordinates):
    """Add the two contact points that replace all the boundary pixels of one pair; returns the next point id."""
    return _add_pair(cleaned_contps, maxPointID, value_cand, value_anta, _pair_geometry(normals, coordinates))


def _contact_pixels(img, first_row=0, first_col=0):
    """(candidates, antagonists, coordinates, normals) of the boundary pixels between two non-zero values of `img`.

    The order is the one of _boundary_directions; `first_row` and `first_col` place a cropped `img` in the full image.
    """
    occupied = img != 0
    right = (img[:, :-1] != img[:, 1:]) & occupied[:, :-1] & occupied[:, 1:]
    below = (img[:-1, :] != img[1:, :]) & occupied[:-1, :] & occupied[1:, :]
    h_rows, h_cols = np.nonzero(right)
    v_rows, v_cols = np.nonzero(below)
    left_value, right_value = img[h_rows, h_cols], img[h_rows, h_cols + 1]
    upper_value, lower_value = img[v_rows, v_cols], img[v_rows + 1, v_cols]
    h_rows, h_cols = (h_rows + first_row).astype(np.float64), (h_cols + first_col).astype(np.float64)
    v_rows, v_cols = (v_rows + first_row).astype(np.float64), (v_cols + first_col).astype(np.float64)
    candidates = np.concatenate((left_value, right_value, upper_value, lower_value)).astype(np.int64)
    antagonists = np.concatenate((right_value, left_value, lower_value, upper_value)).astype(np.int64)
    h_coordinates = np.column_stack((h_cols + 0.5, h_rows))
    v_coordinates = np.column_stack((v_cols, v_rows + 0.5))
    coordinates = np.concatenate((h_coordinates, h_coordinates, v_coordinates, v_coordinates))
    counts = (len(h_rows), len(h_rows), len(v_rows), len(v_rows))
    normals = np.repeat(np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]), counts, axis=0)
    return candidates, antagonists, coordinates, normals


def _pair_geometries(candidates, antagonists, coordinates, normals):
    """{(candidate, antagonist): geometry}; the pixels of a pair stay in the order they come in."""
    pairs = candidates * (int(antagonists.max(initial=0)) + 1) + antagonists
    in_key_order = np.argsort(pairs, kind="stable")
    groups = np.split(in_key_order, np.flatnonzero(np.diff(pairs[in_key_order])) + 1) if pairs.size else []
    return {
        (int(candidates[group[0]]), int(antagonists[group[0]])): _pair_geometry(normals[group], coordinates[group])
        for group in groups
    }


def _wall_contacts(wall):
    """(geometries, values) of a wall image, remembered in this process: every pose of a step adds one stone to it."""
    key = wall.shape, wall.tobytes()
    if key not in WALL_CONTACTS:
        if len(WALL_CONTACTS) >= WALL_CONTACTS_KEPT:
            WALL_CONTACTS.clear()
        WALL_CONTACTS[key] = _pair_geometries(*_contact_pixels(wall)), set(np.unique(wall).tolist())
    return WALL_CONTACTS[key]


def _contact_geometries(img, wall):
    """The geometries of `img`: those of `wall` plus the new pairs, when `img` only fills empty pixels with new values."""
    if wall is not None:
        geometries, wall_values = _wall_contacts(wall)
        changed = img != wall
        new_values = set(np.unique(img[changed]).tolist())
        if not np.any(wall[changed]) and not new_values & wall_values:
            if not new_values:
                return geometries
            rows = np.flatnonzero(changed.any(axis=1))
            cols = np.flatnonzero(changed.any(axis=0))
            first_row, first_col = max(int(rows[0]) - 1, 0), max(int(cols[0]) - 1, 0)
            window = img[first_row : int(rows[-1]) + 2, first_col : int(cols[-1]) + 2]
            pixels = _contact_pixels(window, first_row, first_col)
            added = list(new_values)
            of_new_stone = np.isin(pixels[0], added) | np.isin(pixels[1], added)
            return {**geometries, **_pair_geometries(*(part[of_new_stone] for part in pixels))}
    return _pair_geometries(*_contact_pixels(img))


def contps_of_image(elems, img, wall=None):
    """clear_contps(elems, pixel_to_mesh(img)) without an object for every boundary pixel.

    `wall` is the image before the last stone was added, when there is one: its pairs are not computed again.
    """
    parameters.mu = get_mu()
    geometries = _contact_geometries(img, wall)
    element_order = {key: position for position, key in enumerate(elems)}
    cleaned_contps = dict()
    maxPointID = 0
    # only pairs with contacts, in elems order: contact order decides the LP columns
    for cand, anta in sorted(geometries, key=lambda pair: (element_order[pair[0]], element_order[pair[1]])):
        if elems[cand].id != elems[anta].id:
            maxPointID = _add_pair(cleaned_contps, maxPointID, elems[cand], elems[anta], geometries[(cand, anta)])
    return cleaned_contps


def contact_matrix(elems, img, wall):
    """(contacts, matrix): what solve_force_rigid reads of contps_of_image, and cal_Aglobal of it, built from arrays.

    Every contact is alike for solve_force_rigid, which reads only their number and contact type, so they are one object.
    """
    parameters.mu = get_mu()
    geometries = _contact_geometries(img, wall)
    element_order = {key: position for position, key in enumerate(elems)}
    cand_rows, anta_rows, ends, tangents, normals = [], [], [], [], []
    for cand, anta in sorted(geometries, key=lambda pair: (element_order[pair[0]], element_order[pair[1]])):
        geometry = geometries[(cand, anta)]
        if geometry is not None and elems[cand].id != elems[anta].id:
            cand_rows.append(element_order[cand])
            anta_rows.append(element_order[anta])
            ends.extend(geometry[:2])
            tangents.append(geometry[2])
            normals.append(geometry[3])
    Aglobal = np.zeros((3 * len(elems), 4 * len(cand_rows)))
    if not cand_rows:
        return {}, Aglobal
    centers = np.array([element.center for element in elems.values()])
    coors = np.array(ends)
    t_axes, n_axes = np.repeat(tangents, 2, axis=0), np.repeat(normals, 2, axis=0)
    cols = np.arange(coors.shape[0]) * 2
    for element_rows, sign in ((np.repeat(cand_rows, 2), 1), (np.repeat(anta_rows, 2), -1)):
        rows = element_rows * 3
        t, n = t_axes * sign, n_axes * sign
        R = -coors + centers[element_rows]
        Aglobal[rows, cols] = -1 * t[:, 0]
        Aglobal[rows + 1, cols] = -1 * t[:, 1]
        Aglobal[rows + 2, cols] = -1 * (R[:, 0] * t[:, 1] - R[:, 1] * t[:, 0])
        Aglobal[rows, cols + 1] = -1 * n[:, 0]
        Aglobal[rows + 1, cols + 1] = -1 * n[:, 1]
        Aglobal[rows + 2, cols + 1] = -1 * (R[:, 0] * n[:, 1] - R[:, 1] * n[:, 0])
    contact = ContPoint(0, [0, 0], 0, 0, [0, 0], [0, 0], ContType("friction", parameters))
    return dict.fromkeys(range(coors.shape[0]), contact), Aglobal


def clear_contps(elems, contps):
    parameters.mu = get_mu()
    # make a loop up table for cand, anta -> contact point
    element_order = {key: position for position, key in enumerate(elems)}
    contps_table = dict()
    antas_in_contact = dict()
    for key, value in contps.items():
        if value.anta not in element_order or value.cand not in element_order:
            print(value.coor)
            print(value.orientation)
            raise KeyError((value.cand, value.anta))
        contps_table.setdefault((value.cand, value.anta), []).append(value)
        antas_in_contact.setdefault(value.cand, set()).add(value.anta)

    cleaned_contps = dict()
    maxPointID = 0
    for key_cand, value_cand in elems.items():
        # only pairs with contacts, in elems order: contact order decides the LP columns
        for key_anta in sorted(antas_in_contact.get(key_cand, ()), key=element_order.get):
            value_anta = elems[key_anta]
            if value_cand.id != value_anta.id:
                all_contps = contps_table[(value_cand.id, value_anta.id)]
                if len(all_contps) != 0:
                    normals = np.array([pixel_point.orientation[1] for pixel_point in all_contps], dtype=np.float64)
                    coordinates = np.array([pixel_point.coor for pixel_point in all_contps], dtype=np.float64)
                    maxPointID = _cleaned_pair(cleaned_contps, maxPointID, value_cand, value_anta, normals, coordinates)
    return cleaned_contps
