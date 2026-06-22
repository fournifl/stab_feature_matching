# config/paths.py
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Paths:
    outdir: Path
    matching_technique: 'str'
    matches: Path = field(init=False)
    matches_plot: Path = field(init=False)
    matches_data: Path = field(init=False)
    h: Path = field(init=False)
    warped: Path = field(init=False)

    def __post_init__(self):
        """Initialize subdirectories after creation"""
        self.outdir = self.outdir / self.matching_technique
        self.matches = self.outdir / "matches"
        self.matches_plot = self.matches / "plots"
        self.matches_data = self.matches / "data"
        self.h = self.outdir / "H"
        self.warped = self.outdir / "warped"

    def create_all(self):
        """Create all directories"""
        for path in [self.matches, self.matches_plot, self.matches_data,
                     self.h, self.warped]:
            path.mkdir(parents=True, exist_ok=True)
        return self


# def load_paths(config_path: str = "config.yaml") -> Paths:
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return Paths(Path(config['outdir']))
#
#
# # Singleton
# paths = load_paths()
# paths.create_all()