# src/utils/jupyter.py

import sys
import os
import os
os.environ["OMP_NUM_THREADS"] = "1"

def add_project_root_to_path(root_folder_name="RL-Stock-Market-Masterpiece-Human-x-AI-Collab"):
    current_path = os.path.abspath(os.getcwd())
    while True:
        if os.path.basename(current_path) == root_folder_name:
            if current_path not in sys.path:
                sys.path.insert(0, current_path)
            break
        new_path = os.path.dirname(current_path)
        if new_path == current_path:
            break
        current_path = new_path


add_project_root_to_path()

from src.utils.system import boot

boot()