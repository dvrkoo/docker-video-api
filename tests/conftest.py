import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DummyDlibFace:
    def __init__(self, left, top, right, bottom):
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def left(self):
        return self._left

    def top(self):
        return self._top

    def right(self):
        return self._right

    def bottom(self):
        return self._bottom
