import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, save, output_file
from bokeh.models import Range1d, RangeTool
from bokeh.layouts import column

from stab_fm.core import img

import numpy as np
import cv2


def get_n_matching_pts(ls):
    n = []
    n_valid = []
    t = []
    for f_match in ls:
        df = pd.read_csv(f_match)
        date = img.get_date(f_match)
        t.append(date)
        n.append(len(df))
        n_valid.append(np.sum(df['valid']))
    return t, n, n_valid


def compute_reprojection_error(
        ls_match_pts: list,
        dir_h: list
) -> dict:
    """
    Compute reprojection error between source and destination points
    given a homography matrix H.

    Args:
        src_pts: Source points, shape (N, 2) or (N, 1, 2)
        dst_pts: Destination points, shape (N, 2) or (N, 1, 2)
        H:       3x3 homography matrix (maps src → dst)

    Returns:
        dict with keys:
            - 'errors'      : per-point Euclidean errors, shape (N,)
            - 'mean'        : mean reprojection error
            - 'median'      : median reprojection error
            - 'max'         : max reprojection error
            - 'rmse'        : root mean squared error
            - 'inlier_mask' : bool array, True where error < threshold
    """
    mean = []
    std = []
    max = []
    rmse = []
    t = []

    for f_match in ls_match_pts:

        # read csv
        df = pd.read_csv(f_match)

        # keep only valid (from ransac) matching points
        df = df[df['valid']]

        # extract src and dst points
        src = df[['src_x', 'src_y']].to_numpy()
        dst = df[['dst_x', 'dst_y']].to_numpy()

        # from pathlib import Path
        # f_calib = '/home/florent/Projects/Etretat/Etretat_central2/info/calibration/calib_CAM44.json'
        # dir_im = Path('/home/florent/Projects/Etretat/Etretat_central2/test_stab_fm/target_imgs/')
        # im, _, _, _ = img.read(dir_im / (f_match.stem + '.jpg'), f_calib)
        # f_ref = '/home/florent/Projects/Etretat/Etretat_central2/images/raw/A_Etretat_central2_2fps_600s_20251015_15_00.jpg'
        # im_ref, _, _, _ = img.read(f_ref, f_calib)
        #
        # f, ax = plt.subplots(1, 2)
        # ax[0].imshow(im_ref)
        # ax[0].plot(dst[:, 0], dst[:, 1], c='m', linewidth=0, markersize=6, marker='s')
        # ax[1].imshow(im)
        # ax[1].plot(src[:, 0], src[:, 1], c='gold', linewidth=0, markersize=6, marker='s')
        # plt.show()

        # read h
        f_h = dir_h / (f_match.stem + '.npy')

        if f_h.exists:

            # load homography matrix
            H = np.load(f_h)
            assert H.shape == (3, 3), "H must be a 3x3 matrix"

            # Project src → dst space via H
            projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H)
            projected = projected.reshape(-1, 2)
            # ax[0].plot(projected[:, 0], projected[:, 1], c='g', linewidth=0, markersize=6, marker='s')
            # plt.show()

            # date
            date = img.get_date(f_match)

            # Per-point Euclidean distance
            errors = np.linalg.norm(projected - dst, axis=1)

            # store errors' statistics
            t.append(date)
            mean.append(float(np.mean(errors)))
            std.append(float(np.std(errors)))
            max.append(float(np.max(errors)))
            rmse.append(np.sqrt(np.mean(errors ** 2)))

    return {
          "t" : t,
        "mean": mean,
         "std": std,
         "max": max,
        "rmse": rmse,
    }

def run(paths):

    # list of csv matching points data files
    ls = sorted(paths.matches_data.glob('*.csv'))

    # get number of matching points
    t, n, n_valid = get_n_matching_pts(ls)

    # compute reprojection error
    errors = compute_reprojection_error(ls, paths.h)

    # Range1d objects to share the same ranges between p1 and p2
    x_range = Range1d(min(t), max(t))

    p1 = figure(width=1200, height=200, tools="xpan,xwheel_zoom,reset", title="Number of matching points", x_range=x_range)
    p1.line(t, n, legend_label="raw", line_color="red", line_width=2)
    p1.line(t, n_valid, legend_label="valid", line_color="blue", line_width=2)
    p1.yaxis.axis_label = 'N matching points'

    p2 = figure(width=1200, height=200, tools="xpan,xwheel_zoom,reset", title="Reprojection error", x_range=x_range)
    p2.line(errors['t'], errors['max'], legend_label="max", line_color="chocolate", line_width=2)
    p2.line(errors['t'], errors['mean'], legend_label="mean", line_color="blue", line_width=2)
    p2.line(errors['t'], errors['std'], legend_label="std", line_color="black", line_width=2)
    p2.line(errors['t'], errors['rmse'], legend_label="rmse", line_color="mediumslateblue", line_width=2)
    p2.yaxis.axis_label = 'Reprojection errors (pixels)'

    # singular similarity

    name = 'accuracy_metrics.html'
    output_file(paths.acc_metrics / name)
    layout = column(p1, p2)
    save(layout)

    # plt.plot(t, n, 'r')
    # plt.plot(t, n_valid, 'b')
    # plt.show()


    # get timeseries of the number of raw matches

    # get timeseries of the number of valid (ransac) matches

    # compute reprojection error metrics (mean, std, max)

    # structural_similarity index

    # compute area of matching points

    # goodFeaturesToTrack ?

    return