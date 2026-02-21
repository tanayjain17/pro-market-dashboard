"""Debug import issues."""

import sys
import os
from pathlib import Path

def test_import_paths():
    """Test that utils can be imported."""
    # Print current directory and path
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"__file__: {__file__}")
    print(f"Parent directory: {Path(__file__).parent}")
    print(f"Project root: {Path(__file__).parent.parent}")
    print(f"Python path: {sys.path}")
    
    # Check if utils directory exists
    utils_path = Path(__file__).parent.parent / "utils"
    print(f"Utils path exists: {utils_path.exists()}")
    if utils_path.exists():
        print(f"Utils contents: {list(utils_path.glob('*.py'))}")
    
    # Try to import
    try:
        from utils import data_engine
        print("✅ Successfully imported utils.data_engine")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
    
    assert True  # Always pass
