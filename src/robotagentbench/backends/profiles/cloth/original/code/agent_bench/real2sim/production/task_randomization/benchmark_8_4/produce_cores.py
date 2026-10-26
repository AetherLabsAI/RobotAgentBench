"""Task-owned native PhysX producer; inspect failures before accepting data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from real2sim.production.task8_clear_physx import main

if __name__ == "__main__":
    main("8.4")
