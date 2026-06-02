#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

frequency = __import__('4-frequency').frequency

# Capture matplotlib calls
def test_frequency():
    frequency()

    ax = plt.gca()
    
    # Check labels
    assert ax.get_xlabel() == "Grades", "X label incorrect"
    assert ax.get_ylabel() == "Number of Students", "Y label incorrect"
    assert ax.get_title() == "Project A", "Title incorrect"

    # Check histogram bars
    patches = ax.patches
    assert len(patches) == 10, f"Expected 10 bins, got {len(patches)}"

    # Check edgecolor
    for bar in patches:
        assert bar.get_edgecolor() == (0.0, 0.0, 0.0, 1.0), "Bars must be black outline"

    print("✅ All tests passed!")

test_frequency()
