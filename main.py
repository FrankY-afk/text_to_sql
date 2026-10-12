"""Top-level entry point forwarding to text2sql_engine.main."""

import os
import sys

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from text2sql_engine.main import main

if __name__ == "__main__":
    main()
