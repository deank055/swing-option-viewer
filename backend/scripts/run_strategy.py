# runs only optimal strategy analysis from run_analysis

from run_analysis import compute_data
from swing_option.analysis import optimal_strategy

def main():
    OUTPUT_DIR = "data/output"

    data = compute_data(K_sub=100)
    print("Swing option price = {:.4f}".format(data["option_price"]))

    optimal_strategy.analyse(data, OUTPUT_DIR)

if __name__ == "__main__":
    main()
