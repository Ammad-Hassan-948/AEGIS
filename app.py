"""AEGIS HuggingFace Spaces entry point.

HuggingFace Spaces looks for app.py at the project root.
"""

import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
aegis_dir = os.path.join(root, "aegis")
for d in [root, aegis_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Pre-warm: compile LangGraph before Gradio spawns worker threads
try:
    print("Pre-loading clinical knowledge base and compiling pipeline...", flush=True)
    from graph import aegis_graph  # noqa: F401 — compiles the graph
    print("Pipeline ready.", flush=True)
except Exception as e:
    print(f"Pipeline unavailable (simulation mode will be used): {e}", flush=True)

from ui.gradio_app import create_app

demo = create_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
