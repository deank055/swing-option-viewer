# runs only sensitivity sweep of run_analysis.py
from swing_option.analysis import sensitivity
from run_analysis import compute_data, resolve_k_sub

OUTPUT_DIR = "data/output"

def main():
    K_sub = resolve_k_sub()
    data = compute_data(K_sub=K_sub)
    sensitivity.analyse(data, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()
