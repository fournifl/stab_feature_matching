from roi_editor.core import roi
import cv2
import numpy as np
import matplotlib.pyplot as plt

def masks_from_rois(f_roi, im_shape):

    # read rois
    roi_ = roi.ROICollection(im_shape)
    rois = roi_.load_from_json(f_roi)
    rois = rois.rois

    # compute masks from rois
    masks = [rois[i].compute_mask(im_shape).astype(np.uint8)*255 for i in range(len(rois))]

    return masks


def recover_matching_keypoints(
        ref_fn,
        ref_f_rois,
        target_imgs_dir,
        outdir):

    # read reference image
    im_ref =  cv2.imread(ref_fn)
    im_ref = cv2.cvtColor(im_ref, cv2.COLOR_BGR2RGB)

    # get masks from rois that were defined on ref image
    masks = masks_from_rois(ref_f_rois, im_ref.shape[0:2])

    # compute keypoints and descriptors of ref img
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kps_ref = []
    des_ref = []
    for i in range(len(masks)):
        kp, des = sift.detectAndCompute(im_ref, masks[i])
        kps_ref.append(kp)
        des_ref.append(des)

    # dilate masks for target images
    sz_dil = 800
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz_dil, sz_dil))
    masks_target = [cv2.dilate(masks[i], kernel, iterations=1) for i in range(len(masks))]

    # outdir matches
    outdir_matches = outdir / 'matches'
    outdir_matches.mkdir(parents=True, exist_ok=True)
    outdir_matches_plots = outdir_matches / 'plots'
    outdir_matches_plots.mkdir(parents=True, exist_ok=True)

    # loop through target images
    ls = sorted(target_imgs_dir.glob('*.jp*g'))
    for f in ls:
        print(f)

        # initialize source and destination points
        src_pts = []
        dst_pts = []

        # read target image
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # loop through masks
        for i in range(len(masks)):
            # compute keypoints and descriptors
            kp, des = sift.detectAndCompute(im, masks_target[i])
            raw_matches = bf.knnMatch(des_ref[i], des, k=2)
            good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
            src_pts.append(np.float32([kps_ref[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2))
            dst_pts.append(np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2))

        # combine multiple arrays of shape (n, 1, 2) into a single array of shape (total_n, 1, 2)
        src_pts = np.vstack(src_pts)
        dst_pts = np.vstack(dst_pts)

        # compute homography and inliers
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

        # plot source and destination matches
        src_pts = np.squeeze(src_pts)
        dst_pts = np.squeeze(dst_pts)
        inlier_mask = np.squeeze(inlier_mask)
        inlier_inds = np.where(inlier_mask == 1)
        fig, ax = plt.subplots(1,2, figsize=(22, 12), sharex=True, sharey=True, tight_layout=True)
        ax[0].imshow(im_ref)
        ax[1].imshow(im)
        ax[0].plot(src_pts[:, 0], src_pts[:, 1], c='g', linewidth=0, markersize=6, marker='s')
        ax[1].plot(dst_pts[:, 0], dst_pts[:, 1], c='r', linewidth=0, markersize=6, marker='d')
        ax[1].plot(dst_pts[inlier_inds, 0], dst_pts[inlier_inds, 1], c='b', linewidth=0, markersize=6, marker='d')
        fig.savefig(outdir_matches_plots / f.name)
        # plt.show()

    ### get matching keypoints

    ### filter matching keypoints by a distance criteria

    ### get source and destination keypoints coordinates

    ## subplot source and destination keypoints





    return