#!/usr/bin/env python3
"""
compare_interaction_times.py

Make a stacked-bar comparison (prompt-server vs local processing)
for high- vs low-compute measurements stored in two JSON files.

The script expects:
	1) high-compute JSON path
	2) low-compute  JSON path
optional:
	--out <file>      image file to write (PNG, PDF, etc.)
	--dpi <int>       resolution (default 150)
"""

import json, argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl


# plt.rcParams.update({"font.size": 12})
mpl.rcParams.update({"font.family": "Arial", "font.size": 12.9})
MODES = ["point", "bbox", "scribble", "lasso"]
# right after you define MODES
LABELS = {
	"point":    "Point",
	"bbox":     "Bounding box",   # ← the wording you want
	"scribble": "Scribble",
	"lasso":    "Lasso",
}
SIZES = ["S", "M", "L"]
COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]  # C0–C3


def load_stats(path: Path):
	"""
	Return nested dict stats[size][mode] with means / stds (seconds).
	"""
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	stats = {}
	for size in SIZES:
		stats[size] = {}
		for mode in MODES:
			prompt = np.array(
				[x["server_prompt_processing"] for x in raw[size][mode]], dtype=float
			) / 1000.0  # -> seconds
			local = np.array(
				[x["remainder"] for x in raw[size][mode]], dtype=float
			) / 1000.0
			total = prompt + local
			stats[size][mode] = {
				"prompt_mean": prompt.mean(),
				"prompt_std": prompt.std(ddof=1),
				"local_mean": local.mean(),
				"local_std": local.std(ddof=1),
				"total_mean": total.mean(),
				"total_std": total.std(ddof=1),
			}
	return stats


def build_figure(stats_high, stats_low, out_file: Path, dpi: int):
	fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

	bar_width = 0.18
	x = np.arange(len(SIZES) * 2)  # S-L, M-L, L-L, S-H, M-H, L-H

	# helper to fetch stats per compute tier
	def pick(stats_dict, size, mode, key):
		return stats_dict[size][mode][key]

	for idx_mode, mode in enumerate(MODES):
		# Low-compute segment indices 0-2, High-compute 3-5
		low_means_prompt  = [pick(stats_low,  s, mode, "prompt_mean") for s in SIZES]
		high_means_prompt = [pick(stats_high, s, mode, "prompt_mean") for s in SIZES]
		low_means_local   = [pick(stats_low,  s, mode, "local_mean")  for s in SIZES]
		high_means_local  = [pick(stats_high, s, mode, "local_mean")  for s in SIZES]

		prompt_means = low_means_prompt + high_means_prompt
		local_means  = low_means_local  + high_means_local

		low_std_prompt  = [pick(stats_low,  s, mode, "prompt_std") for s in SIZES]
		high_std_prompt = [pick(stats_high, s, mode, "prompt_std") for s in SIZES]
		low_std_total   = [pick(stats_low,  s, mode, "total_std")  for s in SIZES]
		high_std_total  = [pick(stats_high, s, mode, "total_std")  for s in SIZES]

		prompt_stds = low_std_prompt + high_std_prompt
		total_stds  = low_std_total  + high_std_total

		centers = x + (idx_mode - 1.5) * bar_width

		# plot prompt (bottom, hatched)
		ax.bar(
			centers,
			prompt_means,
			bar_width,
			color=COLOR_CYCLE[idx_mode],
			edgecolor="black",
			hatch="///",
			label=f"{mode} (prompt)" if idx_mode == 0 else None,
			zorder=2,
		)
		# plot local (stacked on top)
		ax.bar(
			centers,
			local_means,
			bar_width,
			bottom=prompt_means,
			color=COLOR_CYCLE[idx_mode],
			edgecolor="black",
			label=mode if idx_mode == 0 else None,
			zorder=2,
		)

		# error bars (prompt left-offset, total right-offset)
		ax.errorbar(
			centers - 0.1 * bar_width,
			prompt_means,
			yerr=prompt_stds,
			fmt="none",
			ecolor="black",
			capsize=1,
			linewidth=1,
			zorder=3,
		)
		ax.errorbar(
			centers + 0.1 * bar_width,
			np.array(prompt_means) + np.array(local_means),
			yerr=total_stds,
			fmt="none",
			ecolor="black",
			capsize=1,
			linewidth=1,
			zorder=3,
		)

	# --- axis cosmetics --------------------------------------------------
	xticks = [rf"Image size: $\mathtt{{{s}}}$" for s in SIZES] * 2
	ax.set_xticks(x)
	ax.set_xticklabels(xticks)
	ax.tick_params(axis="x", length=0)           # hide tick marks

	ax.set_ylabel("Time (s)")
	ax.set_axisbelow(True)
	ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)

	# Horizontal brackets indicating compute tier
	y_line = -0.06  # axes fraction
	text_offset = -0.10
	ax.annotate(
		"",
		xy=(-0.5, y_line),
		xytext=(2.5, y_line),
		xycoords=("data", "axes fraction"),
		arrowprops=dict(arrowstyle="|-|", lw=1, color="black"),
	)
	ax.text(
		1, text_offset,
		r"Computational resources: $\mathtt{L}$",                # only the “L” is monospace
		ha="center", transform=ax.get_xaxis_transform(),
		# fontsize=fontsize
	)
	ax.annotate(
		"",
		xy=(2.5, y_line),
		xytext=(5.5, y_line),
		xycoords=("data", "axes fraction"),
		arrowprops=dict(arrowstyle="|-|", lw=1, color="black"),
	)
	ax.text(
		4, text_offset,
		r"Computational resources: $\mathtt{H}$",                # only the “H” is monospace
		ha="center", transform=ax.get_xaxis_transform(),
		# fontsize=fontsize
	)

	# ---------- legend -------------------------------------------------------
	handles = [
		Patch(
			facecolor=COLOR_CYCLE[i],
			edgecolor="black",
			label=LABELS[MODE]          # use the mapping instead of MODE.capitalize()
		)
		for i, MODE in enumerate(MODES)
	]

	handles += [
		Patch(facecolor="white", edgecolor="black", hatch="///",
			label="Server prompt portion"),
		Patch(facecolor="white", edgecolor="black",
			label="Remaining portion"),
	]
	ax.legend(handles=handles, ncol=2, frameon=True, loc="upper right")

	fig.tight_layout()
	if out_file:
		fig.savefig(out_file, dpi=dpi)
		print(f"Figure written to -> {out_file}")
	else:
		plt.show()


def main():
	parser = argparse.ArgumentParser(description="Compare interaction-time logs.")
	parser.add_argument("high_json", help="High-compute JSON log")
	parser.add_argument("low_json", help="Low-compute  JSON log")
	parser.add_argument("--out", default=None, help="Output image file (PNG, PDF...)")
	parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
	args = parser.parse_args()

	stats_high = load_stats(Path(args.high_json))
	stats_low = load_stats(Path(args.low_json))

	build_figure(stats_high, stats_low, args.out, args.dpi)


if __name__ == "__main__":
	main()


# python plot_timing.py  \
#         high_compute_10_reruns.json   \
#         low_compute_10_reruns.json    \
#         --out figure.png
# python plot_timing.py  \
#         braintumor_highcomp.json    \
#         braintumor_lowcomp.json   \
#         --out speed_measurements.pdf
