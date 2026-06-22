from stab_fm.core import feature_matching

def main(conf):

    feature_matching.feature_based_img_alignment(
        conf.ref_img.fname,
        conf.ref_img.f_rois,
        conf.target_imgs.dir,
        conf.f_calib,
        conf.matching,
        conf.outdir
    )
