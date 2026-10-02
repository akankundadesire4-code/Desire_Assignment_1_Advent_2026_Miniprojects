"""Lets pytest find the modules under src/ without installing a package."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
