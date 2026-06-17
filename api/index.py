import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
aegis_dir = os.path.join(root_dir, "aegis")
for d in [root_dir, aegis_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from aegis.main import app
