from roi_editor.core import roi
import cv2
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, save, output_file
from bokeh.models import Range1d
from bokeh.layouts import row

from stab_fm.core import img

def masks_from_rois(f_roi, im_shape):

    # read rois
    roi_ = roi.ROICollection(im_shape)
    rois = roi_.load_from_json(f_roi)
    rois = rois.rois

    # compute masks from rois
    masks = [rois[i].compute_mask(im_shape).astype(np.uint8)*255 for i in range(len(rois))]

    # compute union of masks
    mask = np.any(masks, axis=0).astype(np.uint8)*255

    return masks, mask


def set_matcher(type_matching):
    if type_matching == 'flann':
        # Match with FLANN (faster than BFMatcher at scale)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif type_matching == 'bfmatcher':
        matcher = cv2.BFMatcher()
    return matcher


def get_matching_pts(matcher, des_ref, des):
    raw_matches = matcher.knnMatch(des_ref, des, k=2)
    return raw_matches


def feature_based_img_alignment(ref_fn, ref_f_rois, target_imgs_dir, f_calib, type_matching, outdir, ecc=False):

    # read reference image
    im_ref, im_ref_gray, h, w = img.read(ref_fn, f_calib)

    # get masks from rois that were defined on ref image
    masks, mask_ref = masks_from_rois(ref_f_rois, (h, w))

    # Create a SIFT object (is an algorithm used to detect and describe local features in images.
    # SIFT is robust to changes in scale, rotation, and illumination)
    sift = cv2.SIFT_create()

    # matcher
    matcher = set_matcher(type_matching) # type_matching = 'flann' or 'bf'

    # compute keypoints and descriptors on ref img
    kps_ref = []
    des_ref = []
    for i in range(len(masks)):
        kp, des = sift.detectAndCompute(im_ref_gray, masks[i])
        kps_ref.append(kp)
        des_ref.append(des)

    # dilate masks for target images
    sz_dil = 800
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz_dil, sz_dil))
    masks_target = [cv2.dilate(masks[i], kernel, iterations=1) for i in range(len(masks))]

    # outdirs
    outdir = outdir / type_matching
    outdir_matches = outdir /  'matches'
    outdir_matches.mkdir(parents=True, exist_ok=True)
    outdir_matches_plots = outdir_matches / 'plots'
    outdir_matches_plots.mkdir(parents=True, exist_ok=True)
    outdir_warped = outdir / 'warped'
    outdir_warped.mkdir(parents=True, exist_ok=True)

    # loop through target images
    ls = sorted(target_imgs_dir.glob('*.jp*g'))
    for f in ls:
        print(f)

        # initialize source and destination points
        dst_pts = []
        src_pts = []

        # read target image
        im, im_gray, _, _ = img.read(f, f_calib)

        # loop through masks applied on target img
        for i in range(len(masks)):
            # compute keypoints and descriptors on target img
            kp, des = sift.detectAndCompute(im_gray, masks_target[i])
            # get matching keypoints
            raw_matches = get_matching_pts(matcher, des_ref[i], des)
            # filter matching keypoints by a distance criteria
            good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
            # append src and dst pts
            dst_pts.append(np.float32([kps_ref[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2))
            src_pts.append(np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2))

        # combine multiple arrays of shape (n, 1, 2) into a single array of shape (total_n, 1, 2)
        dst_pts = np.vstack(dst_pts)
        src_pts = np.vstack(src_pts)

        # compute homography and inliers
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

        # plot source and destination matches (with matplotlib, and bokeh)
        plot_src_and_dst_matches_mpl(src_pts, dst_pts, inlier_mask, im_ref, im, outdir_matches_plots, f.name)
        plot_src_and_dst_matches(src_pts, dst_pts, inlier_mask, im_ref, im, outdir_matches_plots, f.stem, h, w)

        if ecc:
            # ECC
            im_ref_gray = im_ref_gray.astype(np.float32)
            im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
            warp_mode = cv2.MOTION_HOMOGRAPHY
            # The warp matrix is the initial homography, converted to float32 if needed
            warp_matrix = H.astype(np.float32)
            # Termination criteria: stop after 1000 iterations or when epsilon is reached
            eps = 0.001
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, eps)
            # The order of images: (templateImage, inputImage, warpMatrix, ...)
            # It will refine warp_matrix to best align inputImage to templateImage
            mask = cv2.dilate(mask_ref, kernel, iterations=1)
            # cc, refined_h = cv2.findTransformECC(im_ref_gray, im_gray, warp_matrix, warp_mode, criteria, None, mask)
            try:
                cc, refined_H = cv2.findTransformECCWithMask(im_ref_gray, im_gray, mask_ref, mask, warp_matrix, warp_mode, criteria)
                # apply homography to target image
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                warped_img = cv2.warpPerspective(im, refined_H, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # save warped image
                cv2.imwrite(outdir_warped / f.name, warped_img)
            except:
                print('ecc iterations did not converge')
        else:
            # apply homography to target image
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            warped_img = cv2.warpPerspective(im, H, (w, h))
            # save warped image
            cv2.imwrite(outdir_warped / f.name, warped_img)

    return


def plot_src_and_dst_matches_mpl(src_pts, dst_pts, inlier_mask, im_ref, im, outdir_matches_plots, name):
    dst_pts = np.squeeze(dst_pts)
    src_pts = np.squeeze(src_pts)
    inlier_mask = np.squeeze(inlier_mask)
    inlier_inds = np.where(inlier_mask == 1)
    fig, ax = plt.subplots(1, 2, figsize=(22, 12), sharex=True, sharey=True, tight_layout=True)
    ax[0].imshow(im_ref)
    ax[1].imshow(im)
    ax[0].plot(dst_pts[:, 0], dst_pts[:, 1], c='g', linewidth=0, markersize=6, marker='s')
    ax[1].plot(src_pts[:, 0], src_pts[:, 1], c='r', linewidth=0, markersize=6, marker='d')
    ax[1].plot(src_pts[inlier_inds, 0], src_pts[inlier_inds, 1], c='b', linewidth=0, markersize=6, marker='d')
    fig.savefig(outdir_matches_plots / name)


def plot_src_and_dst_matches(src_pts, dst_pts, inlier_mask, im_ref, im, outdir_matches_plots, name, h, w):

    # convert images to rgba
    im_ref = img.to_rgba(im_ref, h, w)
    im = img.to_rgba(im, h, w)

    dst_pts = np.squeeze(dst_pts)
    src_pts = np.squeeze(src_pts)
    inlier_mask = np.squeeze(inlier_mask)
    inlier_inds = np.where(inlier_mask == 1)

    # Bokeh  expects image origin at bottom-left, so flip vertically
    im_ref = np.flipud(im_ref)
    im = np.flipud(im)

    # Flip y-coordinates to match Bokeh's bottom-left origin
    dst_pts[:, 1] = h - dst_pts[:, 1]
    src_pts[:, 1] = h - src_pts[:, 1]

    # Range1d objects to share the same ranges between p1 and p2
    x_range = Range1d(0, w)
    y_range = Range1d(0, h)

    # plot ref img keypoints
    p1 = figure(width=900, height=550, title="Reference image (dst_pts)", x_range=x_range, y_range=y_range)
    p1.image_rgba(image=[im_ref], x=0, y=0, dw=w, dh=h)
    p1.scatter(dst_pts[:, 0], dst_pts[:, 1], color="green", size=6, marker="square")

    # plot moving img keypoints
    p2 = figure(width=900, height=550, title="Current image (src_pts)", x_range=x_range, y_range=y_range)
    p2.image_rgba(image=[im], x=0, y=0, dw=w, dh=h)

    # Outliers in red
    outlier_inds = np.where(inlier_mask == 0)
    p2.scatter(src_pts[outlier_inds, 0].ravel(), src_pts[outlier_inds, 1].ravel(),
               color="red", size=6, marker="diamond")
    # Inliers in blue
    p2.scatter(src_pts[inlier_inds, 0].ravel(), src_pts[inlier_inds, 1].ravel(),
               color="blue", size=6, marker="diamond")

    layout = row(p1, p2)
    name = name + '.html'
    output_file(outdir_matches_plots / name)
    save(layout)



