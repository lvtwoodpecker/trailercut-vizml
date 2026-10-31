"""Shared colors for the static figures, the same tokens as the D3 pages (docs/viz_design.md).

Roles, not hues: MATCHED/POS is the one series color, USED marks shots a trailer took, SPEECH marks
dialogue found in the film, NEG is "less often / away" on diverging plots. Blue, orange and green pass
the colorblind checks together in any combination; don't add a 4th categorical hue to one chart.
"""
from matplotlib.colors import LinearSegmentedColormap

MATCHED = POS = "#2a78d6"   # blue: series 1, matched, "toward / more often"
USED = "#eb6834"            # orange: used in the trailer
SPEECH = "#1baf7a"          # green: dialogue found in the film
NEG = "#e34948"             # red: "away / less often" (diverging pole, never a series color)
WARN = "#eda100"            # amber: a 4th class only where the chart needs it (labels' "same scene")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, GRID = "#c3c2b7", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#c4dbf5", MATCHED, "#123f75"])  # magnitude: one hue, light to dark
