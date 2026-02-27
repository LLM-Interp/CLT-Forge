import os
import sys
from pathlib import Path

GRAPH_PATH = "/path/to/your/attribution_graph.pt"
DICT_PATH = "/path/to/your/autointerp"
PORT = 8200

def main():
    os.environ["ATTR_GRAPH_PATH"] = GRAPH_PATH
    os.environ["DICT_BASE_FOLDER"] = DICT_PATH
    os.environ["PORT"] = str(PORT)

    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src"))

    # from circuit_tracing_visual_interface.app import main as app_main

    # app_main()


if __name__ == "__main__":
    main()
