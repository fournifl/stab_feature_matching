from stab_fm.core import camera_movements
from stab_fm.cli.paths_subdirs_out import Paths

def main(conf):

    path = Paths(conf.outdir, conf.matching)

    camera_movements.run(path.h,
                         conf.ref_img.f_gcps,
                         path.cam_mvts
                         )