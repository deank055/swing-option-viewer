# this file generates some plots used in the calibration chapter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HISTORIC_PATH = "data/input/historic_data_0631.csv"
FUTURES_PATH = "data/input/henryhub_futures_strip_0631.csv"
OUTPUT_PATH = "data/output/henryhub_spot.png"
T0 = pd.Timestamp("2026-06-30")

def load_historic():
    df = pd.read_csv(HISTORIC_PATH)
    df.columns = ["date", "spot"]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.dropna(subset=["spot"])
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = T0 - pd.DateOffset(years=10)
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    return df

def load_forwards():
    df = pd.read_csv(FUTURES_PATH)
    df = df[df["t_years"] > 0].copy()
    df["delivery_date"] = T0 + pd.to_timedelta(df["t_years"] * 365.25, unit="D")
    return df.sort_values("t_years").reset_index(drop=True)

def plot(hist, fwd):
    fig, ax = plt.subplots(figsize=(12, 3))

    for t_start, t_end, label in [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-03-30"), "Winter Storm Uri"),
        (pd.Timestamp("2025-12-01"), pd.Timestamp("2026-02-28"), "Winter Storm Fern"),
    ]:
        mask = (hist["date"] >= t_start) & (hist["date"] <= t_end)
        h = hist[mask]
        ax.fill_between(h["date"], 0, h["spot"], color="#e87722", alpha=0.3, label=label)

    # historical spot
    ax.plot(hist["date"], hist["spot"], color="#1f5fa8", linewidth=0.8, label="Henry Hub spot (historical)")

    # valuation date divider
    ax.axvline(T0, color="#555555", linewidth=1.0, linestyle="--", label="Valuation date (30 Jun 2026)")

    # forward strip
    ax.plot(fwd["delivery_date"], fwd["prior_settle"], color="#d62728", linewidth=1.2, linestyle="--", zorder=4)
    ax.scatter(fwd["delivery_date"], fwd["prior_settle"], color="#d62728", s=10, zorder=5, label="Forward strip (delivery price)")

    # axes
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (USD/MMBtu)", fontsize=11)
    ax.set_ylim(0, 10)

    # date formatting
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.6)

    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")
    # plt.show()

def main():
    hist = load_historic()
    fwd = load_forwards()
    fwd = pd.concat([pd.DataFrame({"delivery_date": [T0], "prior_settle": [hist["spot"].iloc[-1]]}), fwd], ignore_index=True)

    plot(hist, fwd)

def analyse(data, output_dir):
    main()

if __name__ == "__main__":
    main()
