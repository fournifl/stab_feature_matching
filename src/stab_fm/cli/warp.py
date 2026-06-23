from stab_fm.core import warp_imgs
from stab_fm.cli.paths_subdirs_out import Paths

def main(conf):

    paths = Paths(conf.outdir, conf.matching)

    warp_imgs.run(conf.target_imgs.dir,
                  paths.h,
                  paths.warped
                  )

