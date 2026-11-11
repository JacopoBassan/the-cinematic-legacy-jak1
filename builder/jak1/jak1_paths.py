"""Path injection for standalone construction stages and the portable builder."""
import os
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('CINEMATIC_JAK1_DATA', str(Path.cwd()/'data'))).resolve()
WORK = Path(os.environ.get('CINEMATIC_JAK1_WORK', str(Path.cwd()/'build'/'jak1'))).resolve()
