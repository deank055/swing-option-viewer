# this file outputs the models calibrated values and also does the robustness (spoiler, its not very robust)
# 1. used in chapter 5, figures and values
# 2. uses data from the data/input directory

from swing_option.calibration.calibrate import calibrate, load_historic, run_window_stability_analyses, spike_excluded_robustness

HISTORIC_DATA_PATH = "data/input/historic_data_0631.csv"
FUTURE_DATA_PATH = "data/input/henryhub_futures_strip_0631.csv"
OUTPUT_PATH = "data/calibrated_params.json"

def run_calibration_analyses():
   df_full = load_historic(HISTORIC_DATA_PATH)
   t0 = df_full["date"].iloc[-1]

   run_window_stability_analyses(df_full, t0)
   spike_excluded_robustness(HISTORIC_DATA_PATH)

def main():
   print("Calibrating with the following paths:")
   print(f"Historical Data: {HISTORIC_DATA_PATH}")
   print(f"Future Data: {FUTURE_DATA_PATH}")
   print(f"Output Path: {OUTPUT_PATH}")

   # output calibration values
   calibrate(HISTORIC_DATA_PATH, FUTURE_DATA_PATH, OUTPUT_PATH)

   # analyses
   run_calibration_analyses()

if __name__ == "__main__":
    main()
   