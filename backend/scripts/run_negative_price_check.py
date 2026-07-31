# this file checks for negative prices in the trinomial lattice
# 1. Checks whether the negative price clip actually triggers
# 2. Used for numbers in Chapter 4

import os

from swing_option.analysis.clip_diagnostics import run_clip_diagnostics

OUTPUT_DIR = "data/output"

K_SUB_GRID = [1, 4, 8, 16, 32, 64, 100]

def main():
    json_path = os.path.join(OUTPUT_DIR, "clip_diagnostics.json")
    run_clip_diagnostics(K_SUB_GRID, output_path=json_path)

if __name__ == "__main__":
    main()
