"""Build vector figures for the PrefMem evaluation appendices."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


OUT = Path(__file__).resolve().parent

INK = HexColor("#172033")
MUTED = HexColor("#526070")
GRID = HexColor("#CBD5E1")
BLUE = HexColor("#2563EB")
BLUE_LIGHT = HexColor("#DBEAFE")
ORANGE = HexColor("#D97706")
ORANGE_LIGHT = HexColor("#FFEDD5")
GREEN = HexColor("#059669")
GREEN_LIGHT = HexColor("#D1FAE5")
RED = HexColor("#DC2626")
RED_LIGHT = HexColor("#FFE4E6")
PURPLE = HexColor("#7C3AED")


def new_canvas(name: str, width: float, height: float) -> canvas.Canvas:
    c = canvas.Canvas(str(OUT / f"{name}.pdf"), pagesize=(width, height),
                      pageCompression=1)
    c.setTitle(name.replace("-", " ").title())
    return c


def finish(c: canvas.Canvas) -> None:
    c.showPage()
    c.save()


def text(c: canvas.Canvas, x: float, y: float, value: str,
         size: float = 13, bold: bool = False, color=INK,
         align: str = "left") -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "right":
        c.drawRightString(x, y, value)
    elif align == "center":
        c.drawCentredString(x, y, value)
    else:
        c.drawString(x, y, value)


def percent_axis(c: canvas.Canvas, x0: float, x1: float,
                 y0: float, y1: float) -> None:
    for value in (0, 25, 50, 75, 100):
        x = x0 + (x1 - x0) * value / 100
        c.setStrokeColor(GRID)
        c.setLineWidth(0.8)
        c.line(x, y0, x, y1)
        text(c, x, y0 - 18, str(value), 10.5, color=MUTED, align="center")
    text(c, (x0 + x1) / 2, y0 - 34, "Observed rate (%)", 11,
         color=MUTED, align="center")


def grouped_panel(c: canvas.Canvas, top: float, title_value: str,
                  categories: list[str], series: list[tuple[str, list[float], object]],
                  height: float = 170) -> None:
    x0, x1 = 205, 720
    plot_top = top - 32
    plot_bottom = top - height + 42
    text(c, 20, top, title_value, 15, bold=True)

    legend_x = 410
    for name, _, color in series:
        c.setFillColor(color)
        c.rect(legend_x, top - 10, 13, 13, stroke=0, fill=1)
        text(c, legend_x + 19, top - 8, name, 10.5, color=MUTED)
        legend_x += 145

    percent_axis(c, x0, x1, plot_bottom, plot_top)
    group_step = (plot_top - plot_bottom) / len(categories)
    bar_h = min(12, group_step / (len(series) + 0.8))

    for idx, category in enumerate(categories):
        center_y = plot_top - group_step * (idx + 0.5)
        text(c, x0 - 12, center_y - 4, category, 11.5,
             color=INK, align="right")
        offsets = [bar_h * 0.75, -bar_h * 0.75]
        for sidx, (_, values, color) in enumerate(series):
            y = center_y + offsets[sidx] - bar_h / 2
            width = (x1 - x0) * values[idx] / 100
            c.setFillColor(color)
            c.roundRect(x0, y, max(width, 0.5), bar_h, 3,
                        stroke=0, fill=1)
            label_x = min(x0 + width + 7, x1 - 2)
            align = "right" if label_x >= x1 - 3 else "left"
            text(c, label_x, y + 1, f"{values[idx]:.1f}", 10.5,
                 bold=True, color=INK, align=align)


def intent_resolution_results() -> None:
    c = new_canvas("evaluation-intent-results", 760, 640)
    grouped_panel(
        c, 615, "(a) EXP-01: applicable scoped Memory",
        ["Scoped Memory", "Empty Memory", "Exact instruction"],
        [
            ("Autonomous correct", [100.0, 0.0, 100.0], BLUE),
            ("Correct within budget", [100.0, 100.0, 100.0], GREEN),
        ],
        height=190,
    )
    grouped_panel(
        c, 405, "(b) EXP-02: selective autonomy under conflict",
        ["Conflicting", "Unique", "Empty"],
        [
            ("Autonomous correct", [0.0, 97.2, 0.0], BLUE),
            ("Focused clarification", [100.0, 2.8, 97.2], ORANGE),
        ],
        height=190,
    )
    grouped_panel(
        c, 195, "(c) EXP-04: held-out explicit-override replication",
        ["Opposing Memory", "Aligned Memory", "Empty Memory"],
        [
            ("Autonomous composed", [91.7, 100.0, 0.0], BLUE),
            ("Explicit board preserved", [97.2, 100.0, 100.0], GREEN),
        ],
        height=190,
    )
    finish(c)


def memory_lifecycle_results() -> None:
    c = new_canvas("evaluation-memory-lifecycle", 760, 500)
    stages = [
        ("Create initial preference", 8),
        ("Immediate direct recall", 8),
        ("Initial HRI use", 8),
        ("Explicit update", 7),
        ("Updated-value recall", 7),
        ("Updated HRI use", 5),
        ("Delayed direct recall", 2),
        ("Delayed HRI use", 3),
        ("Final direct recall", 2),
    ]
    x0, x1 = 260, 710
    y_top, y_bottom = 445, 55
    for value in range(0, 9, 2):
        x = x0 + (x1 - x0) * value / 8
        c.setStrokeColor(GRID)
        c.setLineWidth(0.8)
        c.line(x, y_bottom, x, y_top)
        text(c, x, 28, str(value), 11, color=MUTED, align="center")
    text(c, (x0 + x1) / 2, 10, "Independent stores meeting the criterion (out of 8)",
         11, color=MUTED, align="center")

    step = (y_top - y_bottom) / len(stages)
    for idx, (stage, value) in enumerate(stages):
        y = y_top - step * (idx + 0.5)
        text(c, x0 - 14, y - 4, stage, 12, align="right")
        width = (x1 - x0) * value / 8
        color = BLUE if idx < 5 else (ORANGE if idx == 5 else RED)
        c.setFillColor(color)
        c.roundRect(x0, y - 8, width, 16, 4, stroke=0, fill=1)
        text(c, x0 + width + 9, y - 4, f"{value}/8", 11.5,
             bold=True)
    finish(c)


def comparison_panel(c: canvas.Canvas, top: float, title_value: str,
                     rows: list[tuple[str, float, float, str, str]],
                     treatment_label: str, control_label: str,
                     height: float) -> None:
    x0, x1 = 260, 720
    plot_top = top - 42
    plot_bottom = top - height + 45
    text(c, 20, top, title_value, 15, bold=True)
    c.setFillColor(BLUE)
    c.rect(430, top - 12, 13, 13, stroke=0, fill=1)
    text(c, 449, top - 10, treatment_label, 10.5, color=MUTED)
    c.setFillColor(ORANGE)
    c.rect(575, top - 12, 13, 13, stroke=0, fill=1)
    text(c, 594, top - 10, control_label, 10.5, color=MUTED)
    percent_axis(c, x0, x1, plot_bottom, plot_top)
    step = (plot_top - plot_bottom) / len(rows)
    for idx, (name, treatment, control, t_count, c_count) in enumerate(rows):
        center_y = plot_top - step * (idx + 0.5)
        text(c, x0 - 12, center_y - 4, name, 11.2, align="right")
        for offset, value, count, color in (
            (7, treatment, t_count, BLUE), (-9, control, c_count, ORANGE)
        ):
            y = center_y + offset - 6
            width = (x1 - x0) * value / 100
            c.setFillColor(color)
            c.roundRect(x0, y, max(width, 0.5), 12, 3, stroke=0, fill=1)
            label_x = min(x0 + width + 7, x1 - 2)
            align = "right" if label_x >= x1 - 3 else "left"
            text(c, label_x, y + 1, count, 10.3, bold=True,
                 align=align)


def monitor_validator_results() -> None:
    c = new_canvas("evaluation-monitor-validator", 760, 650)
    comparison_panel(
        c, 625, "(a) EXP-06: Monitor contribution",
        [
            ("Expected host route", 61.1, 33.3, "22/36", "12/36"),
            ("Visual discrepancy recovery", 20.0, 0.0, "3/15", "0/15"),
            ("Stable endpoint transition", 77.8, 0.0, "7/9", "0/9"),
            ("Visible progress registered", 100.0, 0.0, "3/3", "0/3"),
        ],
        "Production", "No visual", 285,
    )
    comparison_panel(
        c, 315, "(b) EXP-07: Validator contribution and cost",
        [
            ("Expected final route", 97.2, 41.7, "35/36", "15/36"),
            ("False completion", 0.0, 100.0, "0/21", "21/21"),
            ("Correction requested", 100.0, 0.0, "21/21", "0/21"),
            ("Complete state finalized", 93.3, 100.0, "14/15", "15/15"),
        ],
        "Production", "Stop after actions", 285,
    )
    finish(c)


def end_to_end_funnel() -> None:
    c = new_canvas("evaluation-exp09-funnel", 760, 430)
    stages = [
        ("Correct preference-resolved contract", 12),
        ("Correct first compiled action", 12),
        ("Simulator-truth physical goal", 12),
        ("All-strict SAM grounding", 11),
        ("Validator finalized complete", 9),
        ("Exact one-action strict chain", 8),
    ]
    x0, x1 = 290, 710
    y_top, y_bottom = 380, 70
    for value in (0, 3, 6, 9, 12):
        x = x0 + (x1 - x0) * value / 12
        c.setStrokeColor(GRID)
        c.setLineWidth(0.8)
        c.line(x, y_bottom, x, y_top)
        text(c, x, 43, str(value), 11, color=MUTED, align="center")
    text(c, (x0 + x1) / 2, 20, "Episodes (out of 12)", 11,
         color=MUTED, align="center")
    step = (y_top - y_bottom) / len(stages)
    colors = [GREEN, GREEN, GREEN, BLUE, PURPLE, ORANGE]
    for idx, ((stage, value), color) in enumerate(zip(stages, colors)):
        y = y_top - step * (idx + 0.5)
        text(c, x0 - 14, y - 4, stage, 12, align="right")
        width = (x1 - x0) * value / 12
        c.setFillColor(color)
        c.roundRect(x0, y - 10, width, 20, 5, stroke=0, fill=1)
        text(c, x0 + width + 9, y - 4, f"{value}/12", 11.5,
             bold=True)
    finish(c)


def main() -> None:
    intent_resolution_results()
    memory_lifecycle_results()
    monitor_validator_results()
    end_to_end_funnel()


if __name__ == "__main__":
    main()
