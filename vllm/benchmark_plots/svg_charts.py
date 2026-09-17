"""Dependency-free SVG charts for normalized benchmark measurements.

The renderers deliberately know nothing about CSV layouts or file naming.  A
missing point breaks a line; a failed comparison remains a visible heatmap cell.
"""

from __future__ import annotations

import math
import unicodedata
from html import escape
from pathlib import Path


_COLORS = (
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
    "#db2777", "#475569", "#65a30d", "#9333ea",
)
_STATUS_LABELS = {
    "missing": "N/A", "failed": "FAIL", "invalid": "INVALID",
    "zero_baseline": "ZERO",
}


def _units(value: str) -> int:
    """Approximate displayed width, including CJK labels."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value)


def _wrap(value: str, width: int, max_lines: int = 2) -> list[str]:
    value = " ".join(str(value).split())
    if not value:
        return [""]
    lines: list[str] = []
    remaining = value
    while remaining and len(lines) < max_lines:
        used = 0
        end = 0
        for char in remaining:
            size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
            if used + size > width:
                break
            used += size
            end += 1
        if end == len(remaining):
            lines.append(remaining)
            break
        if len(lines) == max_lines - 1:
            lines.append(remaining[:max(1, end - 1)].rstrip() + "…")
            break
        # Prefer a word boundary, but also accommodate paths and long run IDs.
        space = remaining.rfind(" ", 0, end + 1)
        if space > end // 2:
            end = space
        end = max(1, end)
        lines.append(remaining[:end].rstrip())
        remaining = remaining[end:].lstrip()
    return lines


def _text(
    svg: list[str], x: float, y: float, lines: list[str] | str, *,
    size: int = 12, color: str = "#172033", anchor: str = "start",
    weight: int = 400, line_height: int = 17, full: str = "", extra: str = "",
) -> None:
    if isinstance(lines, str):
        lines = [lines]
    svg.append(
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" {extra}>'
    )
    if full:
        svg.append(f"<title>{escape(full)}</title>")
    for index, line in enumerate(lines):
        svg.append(
            f'<tspan x="{x:.2f}" dy="{0 if index == 0 else line_height}">'
            f'{escape(str(line))}</tspan>'
        )
    svg.append("</text>")


def _header(
    width: int, height: int, title: str, subtitle: str, description: str,
) -> tuple[list[str], int]:
    title_lines = _wrap(title, max(20, int((width - 64) / 12)), 2)
    subtitle_lines = _wrap(subtitle, max(20, int((width - 64) / 7)), 3) if subtitle else []
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(description)}</desc>",
        '<style>text { font-family: Inter, "Noto Sans", "Noto Sans CJK SC", '
        'Arial, sans-serif; } .value { paint-order: stroke; stroke: white; '
        'stroke-width: 3px; stroke-linejoin: round; }</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    _text(svg, 32, 38, title_lines, size=22, weight=700, line_height=28, full=title)
    y = 38 + (len(title_lines) - 1) * 28
    if subtitle_lines:
        y += 25
        _text(svg, 32, y, subtitle_lines, size=13, color="#526078", line_height=18, full=subtitle)
        y += (len(subtitle_lines) - 1) * 18
    return svg, y + 28


def _header_height(width: int, title: str, subtitle: str) -> int:
    title_count = len(_wrap(title, max(20, int((width - 64) / 12)), 2))
    subtitle_count = len(_wrap(subtitle, max(20, int((width - 64) / 7)), 3)) if subtitle else 0
    return 66 + (title_count - 1) * 28 + (25 + (subtitle_count - 1) * 18 if subtitle_count else 0)


def _save(svg: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join([*svg, "</svg>"]) + "\n", encoding="utf-8")


def _valid(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 100000 or abs(value) < 0.01:
        return f"{value:.2g}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _axis(maximum: float) -> tuple[float, list[float]]:
    if maximum <= 0:
        return 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    target = maximum * 1.12
    if not math.isfinite(target):
        return maximum, [maximum * (index / 5) for index in range(6)]
    rough = target / 5
    if rough == 0:
        return maximum, [0, maximum]
    magnitude = 10 ** math.floor(math.log10(rough))
    if magnitude == 0:
        return maximum, [0, maximum]
    normalized = rough / magnitude
    factor = next((item for item in (1, 2, 2.5, 5, 10) if normalized <= item), 10)
    step = factor * magnitude
    count = math.ceil(target / step)
    result = step * count
    if not math.isfinite(result):
        return maximum, [maximum * (index / 5) for index in range(6)]
    return result, [index * step for index in range(count + 1)]


def _marker(svg: list[str], x: float, y: float, color: str, shape: int,
            filled: bool = False, detail: str = "") -> None:
    attrs = f'fill="{color if filled else "white"}" stroke="{color}" stroke-width="1.8"'
    svg.append(f'<g class="{"data-point" if detail else "legend-marker"}"><title>{escape(detail)}</title>')
    if shape == 1:
        svg.append(f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" {attrs}/>')
    elif shape in (2, 3):
        offsets = ((0, -5), (4.5, 4), (-4.5, 4)) if shape == 2 else ((0, -5), (5, 0), (0, 5), (-5, 0))
        points = " ".join(f"{x + dx:.2f},{y + dy:.2f}" for dx, dy in offsets)
        svg.append(f'<polygon points="{points}" {attrs}/>')
    else:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" {attrs}/>')
    svg.append('</g>')


def render_lines(
    panels: list[dict], output: Path, *, title: str, subtitle: str,
    y_label: str, linear_x: bool = False, y_scale: str = "auto",
    value_labels: str = "last", columns: int = 2,
) -> None:
    """Draw panels containing ``{label, points: {concurrency: value | None}}``.

    All panels share the full concurrency grid; absent points break each line.
    Auto Y scaling shares an axis when nonempty panel maxima differ by <= 1.6x.
    An all-zero panel uses a safe 0..1 axis when its axis is independent.
    """
    if y_scale not in {"auto", "shared", "independent"}:
        raise ValueError("y_scale must be auto, shared, or independent")
    if value_labels not in {"all", "last", "none"}:
        raise ValueError("value_labels must be all, last, or none")
    if columns < 1:
        raise ValueError("columns must be positive")
    panels = panels or [{"title": "No data", "series": []}]
    grid = sorted({
        concurrency for panel in panels for series in panel["series"]
        for concurrency in series["points"] if concurrency > 0
    })
    labels = sorted({series["label"] for panel in panels for series in panel["series"]})
    styles = {
        label: (_COLORS[index % len(_COLORS)], ("none", "7 4", "2 4")[index // len(_COLORS) % 3])
        for index, label in enumerate(labels)
    }
    strategy_groups = sorted({series["style_group"] for panel in panels
                              for series in panel["series"] if "style_group" in series})
    group_styles = {group: (_COLORS[i % len(_COLORS)], i % 4)
                    for i, group in enumerate(strategy_groups)}
    markers = {}
    for panel in panels:
        for series in panel["series"]:
            if "style_group" in series:
                color, shape = group_styles[series["style_group"]]
                styles[series["label"]] = (color, "none" if series["ep_enabled"] else "8 5")
                markers[series["label"]] = (shape, series["ep_enabled"])
    detailed = bool(strategy_groups)
    values = [
        [value for series in panel["series"] for c, value in series["points"].items()
         if c > 0 and _valid(value)] for panel in panels
    ]
    maxima = [max(items, default=0) for items in values]
    present_maxima = [maximum for maximum, items in zip(maxima, values) if items]
    positive_maxima = [maximum for maximum in present_maxima if maximum > 0]
    if present_maxima and min(present_maxima) == 0 and positive_maxima:
        range_ratio = math.inf
    elif positive_maxima:
        range_ratio = max(positive_maxima) / min(positive_maxima)
    else:
        range_ratio = 1.0
    shared = y_scale == "shared" or (y_scale == "auto" and range_ratio <= 1.6)
    scale_note = "Shared Y-axis" if shared else "Independent Y-axes"
    if y_scale == "auto":
        scale_note += " (auto)"
    x_note = "Linear concurrency axis" if linear_x else "Log2 concurrency axis"

    column_count = 1 if detailed else min(columns, len(panels))
    panel_width = 1240 if detailed else max(560, min(860, 116 + len(grid) * 32))
    plot_width = panel_width - (340 if detailed else 112)
    plot_height = max(420, max(len(p["series"]) for p in panels) * 24) if detailed else 275
    plot_space = plot_height + 90
    width = 32 + column_count * panel_width
    header_height = _header_height(width, title, subtitle)
    layouts: list[dict] = []
    for panel in panels:
        title_lines = _wrap(panel["title"], int((panel_width - 42) / 9), 2)
        longest = max((_units(series["label"]) for series in panel["series"]), default=0)
        legend_columns = 2 if detailed or (longest <= 30 and len(panel["series"]) > 1) else 1
        legend_width = (panel_width - 54) / legend_columns
        entries = [
            _wrap(series["label"], int((legend_width - 35) / 6.7), 2)
            for series in panel["series"]
        ]
        legend_rows = [
            max(len(lines) for lines in entries[start:start + legend_columns]) * 16 + 8
            for start in range(0, len(entries), legend_columns)
        ]
        layouts.append({
            "title_lines": title_lines, "entries": entries,
            "legend_columns": legend_columns, "legend_width": legend_width,
            "legend_rows": legend_rows,
            "heading_height": 27 + 21 * len(title_lines) + sum(legend_rows) + 10,
        })
    row_heights = [
        max(item["heading_height"] for item in layouts[start:start + column_count]) + plot_space
        for start in range(0, len(panels), column_count)
    ]
    height = int(header_height + 28 + sum(row_heights) + 45 + (22 if detailed else 0))
    svg, top = _header(
        width, height, title, subtitle,
        f"{y_label} versus concurrency. {scale_note}. {x_note}. Missing points break lines. "
        "Hover points, titles, and legend labels for full details.",
    )
    _text(svg, 32, top, f"{scale_note} · {x_note} · Gaps indicate unavailable samples", color="#526078")
    if detailed:
        _text(svg, 32, top + 19, "Color / marker: TP / PP / DP · Solid / filled: EP on · Dashed / hollow: EP off",
              color="#526078")
        top += 22
    top += 28
    transformed = [float(value) if linear_x else math.log2(value) for value in grid]
    x_min = min(transformed, default=0)
    x_max = max(transformed, default=0)

    def fraction(concurrency: int) -> float:
        value = float(concurrency) if linear_x else math.log2(concurrency)
        return (value - x_min) / (x_max - x_min) if x_max != x_min else 0.5

    # Dense or uneven grids retain every gridline but show legible tick labels.
    tick_grid: list[int] = []
    tick_spacing = max(44, max((_units(str(c)) * 7 + 8 for c in grid), default=0))
    for concurrency in grid:
        if not tick_grid or (fraction(concurrency) - fraction(tick_grid[-1])) * plot_width >= tick_spacing:
            tick_grid.append(concurrency)
    if grid and grid[-1] not in tick_grid:
        if len(tick_grid) > 1 and (fraction(grid[-1]) - fraction(tick_grid[-1])) * plot_width < tick_spacing:
            tick_grid.pop()
        tick_grid.append(grid[-1])
    row_top = top
    for panel_index, (panel, layout) in enumerate(zip(panels, layouts)):
        row_index, column_index = divmod(panel_index, column_count)
        if column_index == 0 and panel_index:
            row_top += row_heights[row_index - 1]
        panel_left = 16 + column_index * panel_width
        plot_left = panel_left + 80
        plot_right = plot_left + plot_width
        heading_height = row_heights[row_index] - plot_space
        plot_top = row_top + heading_height
        plot_bottom = plot_top + plot_height
        y_max, y_ticks = _axis(max(maxima, default=0) if shared else maxima[panel_index])

        def y_position(value: float) -> float:
            return plot_bottom - (value / y_max) * plot_height

        _text(svg, panel_left + 26, row_top + 22, layout["title_lines"], size=16,
              weight=700, line_height=21, full=panel["title"])
        legend_y = row_top + 27 + len(layout["title_lines"]) * 21
        for index, series in enumerate(panel["series"]):
            legend_row, legend_column = divmod(index, layout["legend_columns"])
            lx = panel_left + 27 + legend_column * layout["legend_width"]
            ly = legend_y + sum(layout["legend_rows"][:legend_row])
            color, dash = styles[series["label"]]
            svg.append(
                f'<line x1="{lx:.1f}" y1="{ly - 4:.1f}" x2="{lx + 21:.1f}" '
                f'y2="{ly - 4:.1f}" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>'
            )
            shape, filled = markers.get(series["label"], (0, False))
            _marker(svg, lx + 10, ly - 4, color, shape, filled)
            _text(svg, lx + 28, ly, layout["entries"][index], line_height=16, full=series["label"])
        svg.append(
            f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" '
            'fill="#f8fafc" stroke="#d7deea"/>'
        )
        for tick in y_ticks:
            y = y_position(tick)
            svg.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#dce3ec"/>')
            _text(svg, plot_left - 9, y + 4, _number(tick), anchor="end", color="#526078")
        for concurrency in grid:
            x = plot_left + fraction(concurrency) * plot_width
            svg.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#e5eaf1"/>')
            if concurrency in tick_grid:
                _text(svg, x, plot_bottom + 22, str(concurrency), anchor="middle", color="#526078")
        center_y = (plot_top + plot_bottom) / 2
        _text(svg, panel_left + 17, center_y, _wrap(y_label, 35, 1), anchor="middle",
              size=13, weight=600, full=y_label,
              extra=f'transform="rotate(-90 {panel_left + 17} {center_y})"')
        _text(svg, (plot_left + plot_right) / 2, plot_bottom + 51, "Concurrency",
              anchor="middle", size=13, weight=600)
        label_positions: dict[int, list[float]] = {}
        end_labels = []
        for series in panel["series"]:
            color, dash = styles[series["label"]]
            segments: list[list[tuple[int, float]]] = [[]]
            for concurrency in grid:
                value = series["points"].get(concurrency)
                if _valid(value):
                    segments[-1].append((concurrency, value))
                elif segments[-1]:
                    segments.append([])
            for segment in segments:
                if len(segment) >= 2:
                    coordinates = " ".join(
                        f"{plot_left + fraction(c) * plot_width:.2f},{y_position(value):.2f}"
                        for c, value in segment
                    )
                    svg.append(
                        f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                        f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" '
                        f'stroke-dasharray="{dash}"/>'
                    )
            series_points = [point for segment in segments for point in segment]
            for concurrency, value in series_points:
                x = plot_left + fraction(concurrency) * plot_width
                y = y_position(value)
                detail = f'{panel["title"]} | {series["label"]} | concurrency={concurrency} | {y_label}={value:g}'
                shape, filled = markers.get(series["label"], (0, False))
                _marker(svg, x, y, color, shape, filled, detail)
            if detailed and series_points:
                c, value = series_points[-1]
                end_labels.append((y_position(value), plot_left + fraction(c) * plot_width,
                                   color, series.get("end_label", series["label"]), value))
            labeled = (series_points if value_labels == "all" else
                       series_points[-1:] if value_labels == "last" and not detailed else [])
            for concurrency, value in labeled:
                x = plot_left + fraction(concurrency) * plot_width
                y = y_position(value)
                positions = label_positions.setdefault(concurrency, [])
                candidates = [max(plot_top + 13, min(plot_bottom - 6, y + delta))
                              for delta in (-10, 17, -26, 33, -42, 49, -58, 65)]
                ly = next((candidate for candidate in candidates
                           if all(abs(candidate - used) >= 13 for used in positions)), candidates[0])
                positions.append(ly)
                anchor, dx = ("start", 6) if x <= plot_left + 35 else ("end", -6) if x >= plot_right - 35 else ("middle", 0)
                _text(svg, x + dx, ly, _number(value), size=10, color=color, anchor=anchor,
                      weight=600, extra='class="value"', full=f"{series['label']}: {value:g}")
        # Reserve a right-hand label column and keep labels at least 23 px apart.
        ordered = sorted(end_labels)
        label_ys = []
        for y, *_ in ordered:
            label_ys.append(max(plot_top + 12, y, label_ys[-1] + 23 if label_ys else plot_top + 12))
        if label_ys and label_ys[-1] > plot_bottom:
            label_ys[-1] = plot_bottom
            for i in range(len(label_ys) - 2, -1, -1):
                label_ys[i] = min(label_ys[i], label_ys[i + 1] - 23)
        for (y, x, color, label, value), ly in zip(ordered, label_ys):
            svg.append(f'<polyline points="{x + 6:.2f},{y:.2f} {plot_right + 12:.2f},{y:.2f} '
                       f'{plot_right + 28:.2f},{ly:.2f}" fill="none" stroke="{color}" '
                       'stroke-width="1" opacity="0.6"/>')
            _text(svg, plot_right + 34, ly + 4, _wrap(label, 31, 1), size=12, color=color,
                  weight=600, full=f"{label}: {value:g}")
        if not values[panel_index]:
            _text(svg, (plot_left + plot_right) / 2, center_y, "No valid data",
                  anchor="middle", size=15, color="#64748b")
    footer = "Hover points for exact values; hover shortened labels for full text."
    if len(tick_grid) < len(grid):
        footer += " Dense concurrency tick labels are sampled."
    _text(svg, 32, height - 22, footer, color="#64748b")
    _save(svg, output)


def _ratio_color(ratio: float, lower_is_better: bool) -> tuple[str, str]:
    # Fixed symmetric log2 scale: 0.25x and 4x have equal color strength.
    signed = -2.0 if ratio == 0 else max(-2.0, min(2.0, math.log2(ratio)))
    strength = abs(signed) / 2
    better = signed < 0 if lower_is_better else signed > 0
    target = (19, 124, 82) if better else (199, 60, 68)
    neutral = (245, 245, 244)
    rgb = tuple(round(start + (end - start) * strength) for start, end in zip(neutral, target))
    linear = [component / 255 / 12.92 if component / 255 <= 0.04045
              else ((component / 255 + 0.055) / 1.055) ** 2.4 for component in rgb]
    luminance = sum(component * weight for component, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
    # Pick black/white by actual contrast, rather than by ratio magnitude: the
    # green and red endpoints have different luminance.
    foreground = "#ffffff" if luminance <= 0.179 else "#000000"
    return "#" + "".join(f"{component:02x}" for component in rgb), foreground


def render_heatmap(
    rows: list[dict], output: Path, *, title: str, subtitle: str,
    lower_is_better: bool,
) -> None:
    """Draw candidate/baseline ratios, retaining unavailable cell statuses.

    Rows contain ``label`` and ``cells: {concurrency: {ratio, status, detail}}``.
    Ratios use a symmetric, saturated log2 color scale centered on one.
    """
    grid = sorted({concurrency for row in rows for concurrency in row["cells"] if concurrency > 0})
    longest = max((_units(row["label"]) for row in rows), default=20)
    label_width = max(260, min(440, int(longest * 6.5) + 26))
    cell_width = max(88, max((len(str(concurrency)) * 8 + 24 for concurrency in grid), default=88))
    width = max(800, 64 + label_width + cell_width * max(1, len(grid)))
    if grid:
        cell_width = (width - 64 - label_width) / len(grid)
    row_labels = [_wrap(row["label"], int((label_width - 30) / 7), 3) for row in rows]
    row_heights = [max(58, len(lines) * 18 + 20) for lines in row_labels]
    header_height = _header_height(width, title, subtitle)
    height = int(header_height + 155 + max(70, sum(row_heights)) + 78)
    direction = "< 1 favors candidate; > 1 favors baseline" if lower_is_better else "> 1 favors candidate; < 1 favors baseline"
    svg, top = _header(
        width, height, title, subtitle,
        f"Candidate / baseline ratio. {direction}. Green is better for the candidate; red is worse. "
        "Missing, failed, invalid, and zero-baseline comparisons retain status labels.",
    )
    _text(svg, 32, top, f"Candidate / baseline · {direction}", size=13, weight=600)
    _text(svg, 32, top + 22, "Green: candidate better · Red: candidate worse · Neutral: equal · Log2 color scale, saturated at 0.25× / 4×",
          size=12, color="#526078")
    legend_x = 32
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        color, foreground = _ratio_color(ratio, lower_is_better)
        svg.append(f'<rect x="{legend_x}" y="{top + 36}" width="68" height="29" rx="4" fill="{color}"/>')
        _text(svg, legend_x + 34, top + 55, f"{ratio:g}×", anchor="middle", color=foreground, weight=600)
        legend_x += 73
    table_top = top + 124
    cell_left = 32 + label_width
    _text(svg, 32, table_top - 14, "Series / candidate", size=12, color="#526078", weight=600)
    _text(svg, cell_left + (width - 32 - cell_left) / 2, table_top - 42,
          "Concurrency", anchor="middle", size=13, weight=600)
    for index, concurrency in enumerate(grid):
        _text(svg, cell_left + (index + 0.5) * cell_width, table_top - 14,
              str(concurrency), anchor="middle", color="#526078", weight=600)
    y = table_top
    for row, lines, row_height in zip(rows, row_labels, row_heights):
        _text(svg, 32, y + row_height / 2 - (len(lines) - 1) * 9 + 4,
              lines, size=12, line_height=18, full=row["label"])
        for index, concurrency in enumerate(grid):
            cell = row["cells"].get(concurrency, {"status": "missing", "ratio": None, "detail": "No sample"})
            status = cell.get("status", "missing")
            ratio = cell.get("ratio")
            if status == "ok" and _valid(ratio):
                fill, foreground = _ratio_color(ratio, lower_is_better)
                label = _number(ratio) + "×"
            else:
                fill, foreground = "#f1f5f9", "#64748b"
                label = _STATUS_LABELS.get(status, "INVALID")
                if status == "failed":
                    fill, foreground = "#ffedd5", "#9a3412"
            x = cell_left + index * cell_width
            detail = f'{row["label"]} | concurrency={concurrency} | status={status}'
            if ratio is not None:
                detail += f" | candidate/baseline={ratio}"
            if cell.get("detail"):
                detail += f" | {cell['detail']}"
            svg.append(f'<g><title>{escape(detail)}</title>')
            svg.append(
                f'<rect x="{x + 2:.2f}" y="{y + 2:.2f}" width="{cell_width - 4:.2f}" '
                f'height="{row_height - 4:.2f}" rx="5" fill="{fill}"/>'
            )
            _text(svg, x + cell_width / 2, y + row_height / 2 + 5, label,
                  anchor="middle", size=14 if label.endswith("×") else 11,
                  color=foreground, weight=600)
            svg.append("</g>")
        y += row_height
    if not rows or not grid:
        _text(svg, width / 2, table_top + 38, "No comparison data", anchor="middle", size=15, color="#64748b")
    _text(svg, 32, height - 42, "N/A: missing sample · FAIL: failed run · INVALID: unavailable metric · ZERO: baseline is zero",
          size=12, color="#526078")
    _text(svg, 32, height - 21, "Hover cells for exact values, status details, and full series names.", size=12, color="#64748b")
    _save(svg, output)
