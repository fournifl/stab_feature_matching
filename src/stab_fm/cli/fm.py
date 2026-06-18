from stab_fm.core import feature_matching

def main(conf):

    feature_matching.recover_matching_keypoints(
        conf.ref_img.fname,
        conf.ref_img.f_rois,
        conf.target_imgs.dir,
        conf.outdir
    )
