from stab_fm.core import feature_matching
from stab_fm.cli.paths import Paths

def main(conf):

    feature_matching.run(
        conf.ref_img.fname,
        conf.ref_img.f_rois,
        conf.target_imgs.dir,
        conf.f_calib,
        conf.matching,
        Paths(conf.outdir, conf.matching)
    )
