from stab_fm.core import accuracy_metrics
from stab_fm.cli.paths import Paths

def main(conf):
    accuracy_metrics.run(
        Paths(conf.outdir, conf.matching)
    )
