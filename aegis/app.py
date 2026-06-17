"""AEGIS — entry point.

Run with:   python app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-warm the environment and compile the LangGraph in the main thread.
# This must happen before Gradio spawns any worker/daemon threads.
print("Pre-loading clinical knowledge base...", flush=True)

print("Compiling LangGraph pipeline...", flush=True)
from graph import aegis_graph  # noqa: F401 — side-effect: compiles the graph

print("Pre-loading complete. Starting Gradio...", flush=True)

from ui.gradio_app import create_app, CSS

# Create the Gradio app
demo = create_app()

# Expose the underlying FastAPI app for serverless environments (like Vercel)
app = demo.app

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
    )
