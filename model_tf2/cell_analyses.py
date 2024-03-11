import copy as cp
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.transform import resize
from scipy import ndimage


def shift_data(spikes):
    perm = np.random.randint(low=-200, high=200)
    spikes_perm = cp.deepcopy(spikes)
    spikes_perm = np.roll(spikes_perm, perm, axis=1)
    return spikes_perm


def get_surrounding_bins(rate_map, obj_loc, width):
    vecs = []
    for loc in obj_loc:
        # choose 8 surrounding states
        pos_ = [loc - 1, loc + 1, loc + width, loc - width, loc + width - 1, loc - width - 1, loc + width + 1,
                loc - width + 1]
        vec = [rate_map[:, p] for p in pos_]
        vecs.append(np.asarray(vec).T)

    return vecs


def auto_rotate(auto, angle):
    auto_rot = rotate(auto, angle)

    return auto_rot


def get_circle(cell, x, y, radius, ring=False):
    circle = []

    rows = [y + p for p in range(-int(radius) + 1, int(radius))]
    cols = [x + p for p in range(-int(radius) + 1, int(radius))]
    for row in rows:
        row_sq = (row - y) ** 2
        for col in cols:
            col_sq = (col - x) ** 2
            cond = (radius / 2) ** 2 < row_sq + col_sq < radius ** 2 if ring else row_sq + col_sq < radius ** 2
            if cond:
                circle.append(cell[row, col])
    return circle


def get_corr_rot(auto, angle, radius, ring=False):
    auto_rot = auto_rotate(auto, angle)

    # x, y = np.where(auto == np.max(auto))
    # x, y = x[0], y[0]
    y, x = [int((x - 1) / 2) for x in auto.shape]

    # x_1, y_1 = np.where(auto_rot == np.max(auto_rot))
    # x_1, y_1 = x_1[0], y_1[0]
    y_1, x_1 = [int((x - 1) / 2) for x in auto_rot.shape]

    c1 = get_circle(auto, x, y, radius, ring=ring)
    c2 = get_circle(auto_rot, x_1, y_1, radius, ring=ring)

    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(auto, cmap='jet')
    plt.title(str(y) + ' ' + str(x))

    plt.figure()
    plt.imshow(auto_rot, cmap='jet')
    plt.title(str(y_1) + ' ' + str(x_1))
    """

    return np.corrcoef(c1, c2)[0, 1]


def get_grid_score(auto, radius, ring=False):
    angles_1 = [60, 120]
    angles_2 = [30, 90, 150]

    corrs_1, corrs_2 = [], []
    for angle in angles_1:
        corrs_1.append(get_corr_rot(auto, angle, radius, ring=ring))

    for angle in angles_2:
        corrs_2.append(get_corr_rot(auto, angle, radius, ring=ring))

    return min(corrs_1) - max(corrs_2)


def detect_peaks(image, thresh=None):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    if thresh is not None:
        image = cp.deepcopy(image)
        # breakpoint()
        image[image < thresh * np.max(image)] = 0.0

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    y_peaks, x_peaks = np.where(detected_peaks)

    return detected_peaks, y_peaks, x_peaks


def get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim):
    # detected_peaks = detect_peaks(auto)

    # x_c, y_c = np.where(auto == np.max(auto))
    # y_c, x_c = y_c[0], x_c[0]
    # x, y = np.where(detected_peaks)

    dists = []
    for b, a in zip(y_peaks, x_peaks):
        dists.append(np.sqrt((x_c - a) ** 2 + (y_c - b) ** 2))
    dists.sort()

    try:
        radius = np.mean(dists[5:7])
    except IndexError:
        radius = 8

    # print('get_radius', radius, radius_lim)

    if np.isnan(radius):  # if no peaks
        radius = 8
    if radius >= radius_lim:
        radius = int(radius_lim * 0.75)

    return radius


def get_final_radius(auto, radius_lim):
    # y_c, x_c = np.where(auto == np.max(auto))
    # y_c, x_c = y_c[0], x_c[0]
    y_c, x_c = [int((x - 1) / 2) for x in auto.shape]

    detected_peaks, y_peaks, x_peaks = detect_peaks(auto)
    radius = get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim)

    field_size = get_field_size(auto, y_c, x_c)
    radius = np.ceil(radius + field_size)

    if radius >= radius_lim:
        radius = int(radius_lim * 0.75)

    return radius


def get_6_clostest_peaks(auto, radius, thresh=None):
    # y_c, x_c = np.where(auto == np.max(auto))
    # y_c, x_c = y_c[0], x_c[0]
    y_c, x_c = [int((x - 1) / 2) for x in auto.shape]

    _, y_peaks, x_peaks = detect_peaks(auto, thresh=thresh)
    # radius = get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim)
    # find x, y coordinates of 6 closest peaks
    y_x_ = []
    dists = []
    for b, a in zip(y_peaks, x_peaks):
        dist = np.sqrt((x_c - a) ** 2 + (y_c - b) ** 2)
        if 0 < dist <= radius:
            y_x_.append([b, a])
            dists.append(dist)
    # sort by distance
    idx = np.argsort(dists)
    idx = idx[:6]
    if len(dists) > 0:
        return [y_x_[a] for a in idx], np.mean(np.sort(dists)[:6])
    else:
        return [], np.nan


def get_field_size(auto, y_c, x_c):
    # central field size
    # x_c, y_c = np.where(auto == np.max(auto))
    # y_c, x_c = y_c[0], x_c[0]

    y_f, x_f = np.where(auto <= auto[y_c, x_c] / 2)
    dists_f = []
    for b, a in zip(y_f, x_f):
        dists_f.append(np.sqrt((x_c - a) ** 2 + (y_c - b) ** 2))
    dists_f.sort()

    if len(dists_f) > 0:
        field_size = dists_f[0]

        return field_size
    else:
        return 0.0


def elliptical_fit(y_x_):
    a = np.zeros((5, 5))
    skip_rand = np.random.randint(0, 6)
    counter = 0
    for i, (y, x) in enumerate(y_x_):
        if i == skip_rand:
            continue
        # x = x - x_c
        # y = y - y_c
        a[counter, 0] = x ** 2
        a[counter, 1] = x * y
        a[counter, 2] = y ** 2
        a[counter, 3] = x
        a[counter, 4] = y
        counter += 1

    f = 1
    a, b, c, d, e = np.dot(np.matmul(np.linalg.inv(np.matmul(a.T, a)), a.T), f * np.array([1, 1, 1, 1, 1]))
    # print('A, B, C, D, E : ',  A, B, C, D, E)
    discrim = b * b - 4 * a * c
    # print('discriminat : ', discrim)
    m1 = 2 * (a * e * e + c * d * d - b * d * e + discrim * f)
    semi_major = -np.sqrt(m1 * (a + c + np.sqrt((a - c) ** 2 + b * b))) / discrim
    semi_minor = -np.sqrt(m1 * (a + c - np.sqrt((a - c) ** 2 + b * b))) / discrim

    theta = np.arctan((c - a - np.sqrt((a - c) ** 2 + b * b)) / b)

    return theta, semi_major, semi_minor


def ellipse_correct(auto, theta, semi_major, semi_minor):
    auto_rot = auto_rotate(auto, +theta * 180 / np.pi)

    if np.abs(semi_minor) <= 1e-8 or np.isnan(semi_minor):
        raise ValueError('semi minor axis is nan or too small')
    if np.isnan(semi_major) or np.abs(semi_major) > 10000:
        raise ValueError('semi major axis is nan or too big')
    ratio = semi_major / semi_minor

    y, x = np.shape(auto_rot)
    auto_rot = resize(auto_rot, (int(y * ratio), x))
    auto_correct = auto_rotate(auto_rot, -theta * 180 / np.pi)

    # print(np.shape(auto_correct), np.shape(auto))

    y_diff, x_diff = [np.maximum(int((x - y) / 2), 1) for x, y in zip(np.shape(auto_correct), np.shape(auto))]
    # print(y_diff, x_diff, 'poo')
    auto_correct = auto_correct[y_diff:-y_diff, x_diff:-x_diff]
    return auto_correct


def grid_score_scale_analysis(auto, fit_ellipse=True, ring=False):
    theta = np.nan
    # find scale limit guess
    radius_lim = min(np.shape(auto)) / 2 + 5
    # get center of auto
    y_c, x_c = [int((x - 1) / 2) for x in auto.shape]
    # detect local peaks
    detected_peaks, y_peaks, x_peaks = detect_peaks(auto, thresh=0.3)
    # plt.scatter(x_peaks, y_peaks, c='w', s=3)
    # get spatial scale
    radius = get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim) + get_field_size(auto, y_c, x_c)
    # get 6 local peaks
    y_x_, grid_scale = get_6_clostest_peaks(auto, radius, thresh=0.3)
    # print(y_x_)
    # fit an ellipse
    if fit_ellipse:
        try:
            theta, semi_major, semi_minor = elliptical_fit(y_x_)
            # correct for ellipse
            auto_correct = ellipse_correct(auto, theta, semi_major, semi_minor)
            # print(str(i))
            if np.min(auto_correct.shape) == 0:
                raise ValueError('auto got squished to zero')
        except (np.linalg.LinAlgError, ValueError, OverflowError):  # as e
            # plt.title('n/a')
            # print(str(i) + ' didnt fit ellipse because of ' + str(e))
            return np.nan, np.nan, np.nan

        # find scale limit guess
        radius_lim = max(np.shape(auto_correct)) / 2 + 5
        # get new center
        y_c, x_c = [int((x - 1) / 2) for x in auto_correct.shape]
        # get peaks
        detected_peaks, y_peaks, x_peaks = detect_peaks(auto_correct, thresh=0.3)
        # find scale
        radius = get_radius(y_peaks, x_peaks, y_c, x_c, radius_lim) + get_field_size(auto_correct, y_c, x_c)
    else:
        auto_correct = auto

    g_s_possible = []
    for radius_ in [radius - 6, radius - 5, radius - 4, radius - 3, radius - 2, radius - 1, radius, radius + 1,
                    radius + 2, radius + 3]:
        try:
            grid_score = get_grid_score(auto_correct, radius_, ring=ring)
            g_s_possible.append(grid_score)
        except IndexError:
            pass
    if len(g_s_possible) > 0:
        grid_score = np.nanmax(g_s_possible)
        return grid_score, grid_scale, theta
        # plt.title(str(grid_score)[:5] + '  ' + str(grid_scale)[:4])
    else:
        # plt.title('n/a')
        return np.nan, np.nan, np.nan


def calc_entropy(pc):
    return - np.sum(pc * np.log(pc))


def norm_pc(pc):
    pc = (pc - pc.min()) / (pc.max() - pc.min() + 1e-12)
    pc = pc + 1e-10
    pc = pc / np.sum(pc)
    return pc


def find_biggest_connected_component(pc):
    labeled, nr_objects = ndimage.label(pc > pc.max()/5)
    # find biggest connected component mass
    masses = []
    for i in range(nr_objects):
        masses.append(np.sum(pc[labeled == i + 1]))

    try:
        index = np.argmax(masses) + 1
    except ValueError:
        return np.nan, np.nan, np.nan
    mass_frac = max(masses) / np.sum(pc)

    return labeled, index, mass_frac


def remove_biggest_connected_component(pc, labeled, index):
    # remove that part of place cell
    labeled[labeled != index] = 0.0
    labeled[labeled > 0.0] = 1.0
    pc = pc - pc * labeled

    return pc


def place_cell_metric(pc):
    if pc.max() == pc.min():
        return np.nan
    # import matplotlib.pyplot as plt
    # entropy_max = calc_entropy(np.ones_like(pc) / pc.size)

    pc = norm_pc(pc)
    # entropy_before = calc_entropy(pc)

    # plt.imshow(pc)
    # plt.colorbar()
    # plt.show()

    labeled, index, mass_frac = find_biggest_connected_component(pc)

    return mass_frac
    # pc_rem = remove_biggest_connected_component(pc, labeled, index)
    # pc_rem = norm_pc(pc_rem)
    # entropy_after = calc_entropy(pc_rem)
    # plt.imshow(pc)
    # plt.colorbar()
    # plt.show()
    # metric = (entropy_after - entropy_before) / entropy_max
    # return metric,  (entropy_before, entropy_after, entropy_max)
