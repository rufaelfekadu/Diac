#!/usr/bin/env python3
"""Collect DER metrics from evaluation log files and summarize into a table.

It scans under a results directory (default: ./results) for log files that match:
  results/<model+type>/<training_dataset>/logs/*.log

Each log file is expected to contain sections like:
  [INFO]   Model: lstm
  [INFO]   Model Type: text-only
  [INFO]   Dataset: tashkeela
  [INFO]   Test file: data/clartts/test.txt

And a DER table shaped like:
  |       |  With case ending  | Without case ending |  With case ending  | Without case ending |
  |  DER  | ...
  |   %   |    6.43    |    4.95    |    7.89    |    6.03    |

We extract the four DER values in order:
  der_with_case_incl_no_diacritic
  der_without_case_incl_no_diacritic
  der_with_case_excl_no_diacritic
  der_without_case_excl_no_diacritic

Output formats:
    CSV (default), Markdown table, or LaTeX tabular.

Header grouping:
    - CSV: two header rows; DER columns are grouped under "Including no_diac" and "Excluding no_diac" with subheaders "With case" and "Without case".
    - Markdown: uses an HTML table with thead and colspan to render grouped headers.
    - LaTeX: uses \\multicolumn and \\cmidrule (booktabs) to group the DER columns.

Extremes highlighting (optional):
    Use flags to annotate lowest/highest/second-highest values per metric column:
        --mark-lowest, --mark-highest, --mark-second-highest
    CSV appends markers like "[min]", "[max]", "[2nd]" to values.
    Markdown uses inline styles on cells (green=min, red=max, amber=2nd).
    LaTeX uses text formatting (\\textbf=min, \\textit=max, \\underline=2nd).

Usage examples:
  python scripts/collect_der.py
  python scripts/collect_der.py --results_dir results --format markdown
  python scripts/collect_der.py --output der_summary.csv

Columns in output:
  model_full,model,model_type,training_dataset,eval_set,log_path,
  der_w_case_incl_no_diac,der_wo_case_incl_no_diac,der_w_case_excl_no_diac,der_wo_case_excl_no_diac
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DER_VALUES_RE = re.compile(r"^\|\s*%\s*\|(.+?)\|\s*$")
WER_VALUES_RE = re.compile(r"^\|\s*(?:%|WER)\s*\|(.+?)\|\s*$", re.IGNORECASE)
MODEL_RE = re.compile(r"Model:\s*(\S+)")
MODEL_TYPE_RE = re.compile(r"Model Type:\s*(\S+)")
DATASET_RE = re.compile(r"Dataset:\s*(\S+)")
TEST_FILE_RE = re.compile(r"Test file:\s*(\S+)")


@dataclass
class DerRecord:
    model_full: str
    model: str
    model_type: str
    training_dataset: str
    eval_set: str
    log_path: str
    der_w_case_incl_no_diac: Optional[float]
    der_wo_case_incl_no_diac: Optional[float]
    der_w_case_excl_no_diac: Optional[float]
    der_wo_case_excl_no_diac: Optional[float]

    def to_row(self, columns: List[str]) -> List[str]:
        data = asdict(self)
        out = []
        for c in columns:
            v = data.get(c)
            if v is None:
                out.append("")
            else:
                out.append(f"{v}")
        return out

def iter_log_files(results_dir: Path) -> Iterable[Path]:
    # Pattern: results/<model>/<dataset>/logs/eval-*.log
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            logs_dir = dataset_dir / "logs"
            if not logs_dir.is_dir():
                continue
            for log_file in logs_dir.glob("eval-*.log"):
                yield log_file

def parse_der_line(line: str) -> Optional[List[float]]:
    m = DER_VALUES_RE.match(line)
    if not m:
        return None
    # Extract numeric tokens between pipe separators.
    inner = m.group(1)
    # Split on '|' and filter tokens containing a digit.
    parts = [p.strip() for p in inner.split('|')]
    nums: List[float] = []
    for p in parts:
        # Accept numbers like 6.43 or 6 or 6,43 (replace comma)
        p2 = p.replace(',', '.')
        if re.search(r"\d", p2):
            try:
                nums.append(float(p2))
            except ValueError:
                continue
    if len(nums) >= 4:
        return nums[:4]
    return None

def parse_log(log_path: Path) -> Optional[DerRecord]:
    model = model_type = training_dataset = eval_set = None
    der_values = None
    expecting_der_block = False
    try:
        with log_path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_stripped = line.strip()
                if 'Model:' in line_stripped and model is None:
                    mm = MODEL_RE.search(line_stripped)
                    if mm:
                        model = mm.group(1)
                elif 'Model Type:' in line_stripped and model_type is None:
                    mt = MODEL_TYPE_RE.search(line_stripped)
                    if mt:
                        model_type = mt.group(1)
                elif 'Dataset:' in line_stripped and training_dataset is None:
                    ds = DATASET_RE.search(line_stripped)
                    if ds:
                        training_dataset = ds.group(1)
                elif 'Test file:' in line_stripped and eval_set is None:
                    tf = TEST_FILE_RE.search(line_stripped)
                    if tf:
                        # Expect format like data/clartts/test.txt
                        parts = Path(tf.group(1)).parts
                        # Find 'data' then next part
                        if 'data' in parts:
                            idx = parts.index('data')
                            if idx + 1 < len(parts):
                                eval_set = parts[idx + 1]
                # Identify DER section
                if re.search(r"\|\s+DER\s+\|", line):
                    expecting_der_block = True
                elif expecting_der_block and '|   %' in line:
                    vals = parse_der_line(line)
                    if vals:
                        der_values = vals
                        expecting_der_block = False
                        # We can break after capturing DER to speed up
                        # but continue parsing to pick up any missing metadata
                elif expecting_der_block and line.strip().startswith('+'):
                    # End of table before we found % row
                    expecting_der_block = False
    except Exception as e:
        print(f"Warning: failed to parse {log_path}: {e}")
        return None

    # Fallbacks from path if missing
    if (model is None or model_type is None or training_dataset is None) and 'results' in log_path.parts:
        # results/<model_full>/<training_dataset>/logs/file
        try:
            idx = log_path.parts.index('results')
            model_full_dir = log_path.parts[idx + 1]
            path_training_dataset = log_path.parts[idx + 2]
            if model is None or model_type is None:
                # Attempt to split model_full_dir by last hyphen grouping known model types? We'll just leave as model_full if can't split.
                if model is None and model_type is None and '-' in model_full_dir:
                    # heuristic: last two tokens maybe denote type (e.g., text-only)
                    # Use first token as model, rest as type
                    tokens = model_full_dir.split('-')
                    model = tokens[0]
                    model_type = '-'.join(tokens[1:])
                else:
                    model = model or model_full_dir
                    model_type = model_type or 'unknown'
            training_dataset = training_dataset or path_training_dataset
        except Exception:
            pass
    if eval_set is None:
        eval_set = 'unknown'

    if model is None:
        return None
    if model_type is None:
        model_type = 'unknown'
    if training_dataset is None:
        training_dataset = 'unknown'

    model_full = f"{model}-{model_type}" if model_type and not model.endswith(model_type) else model

    dv = der_values or [None, None, None, None]
    return DerRecord(
        model_full=model_full,
        model=model,
        model_type=model_type,
        training_dataset=training_dataset,
        eval_set=eval_set,
        log_path=str(log_path),
        der_w_case_incl_no_diac=dv[0],
        der_wo_case_incl_no_diac=dv[1],
        der_w_case_excl_no_diac=dv[2],
        der_wo_case_excl_no_diac=dv[3],
    )

def collect(records: Iterable[DerRecord]) -> List[DerRecord]:
    # Optionally could de-duplicate by (model_full, training_dataset, eval_set)
    # For now just return sorted list
    return sorted(records, key=lambda r: (r.model_full, r.training_dataset, r.eval_set, r.log_path))


COLUMNS = [
    'model_full', 'model', 'model_type', 'training_dataset', 'eval_set', 'log_path',
    'der_w_case_incl_no_diac', 'der_wo_case_incl_no_diac', 'der_w_case_excl_no_diac', 'der_wo_case_excl_no_diac'
]

# Metric column names (subset of COLUMNS)
METRIC_COLUMNS = COLUMNS[6:]

ExtremesMap = Dict[str, Dict[str, Set[int]]]

def compute_extremes(records: List[DerRecord]) -> ExtremesMap:
    """Compute row indices for min, max, and second-highest per metric column.

    Returns a mapping: { column_name: { 'min': set(rows), 'max': set(rows), 'second_max': set(rows) } }
    """
    extremes: ExtremesMap = {col: {'min': set(), 'max': set(), 'second_max': set()} for col in METRIC_COLUMNS}
    # For each metric, collect (row_idx, value)
    for col in METRIC_COLUMNS:
        pairs: List[Tuple[int, float]] = []
        for idx, rec in enumerate(records):
            v = getattr(rec, col)
            if v is None:
                continue
            try:
                pairs.append((idx, float(v)))
            except Exception:
                continue
        if not pairs:
            continue
        values = [v for _, v in pairs]
        v_min = min(values)
        v_max = max(values)
        # Determine second-highest unique value
        uniq_desc = sorted(set(values), reverse=True)
        v_second_max: Optional[float] = uniq_desc[1] if len(uniq_desc) > 1 else None
        for idx, v in pairs:
            if v == v_min:
                extremes[col]['min'].add(idx)
            if v == v_max:
                extremes[col]['max'].add(idx)
            if v_second_max is not None and v == v_second_max:
                extremes[col]['second_max'].add(idx)
    return extremes

def output_csv(records: List[DerRecord], out_file: Optional[Path], extremes: Optional[ExtremesMap] = None,
               mark_lowest: bool = False, mark_highest: bool = False, mark_second: bool = False):
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        f = out_file.open('w', newline='', encoding='utf-8')
    else:
        import sys
        f = sys.stdout
    with f:
        writer = csv.writer(f)
        # Grouped header: two rows. First row shows group titles for DER columns.
        base_cols = [
            'model_full', 'model', 'model_type', 'training_dataset', 'eval_set', 'log_path'
        ]
        group_header = [''] * len(base_cols) + [
            'Including no_diac', '', 'Excluding no_diac', ''
        ]
        sub_header = base_cols + [
            'With case', 'Without case', 'With case', 'Without case'
        ]
        writer.writerow(group_header)
        writer.writerow(sub_header)
        for row_idx, r in enumerate(records):
            row = r.to_row(COLUMNS)
            if extremes is not None and (mark_lowest or mark_highest or mark_second):
                # Annotate metric cells
                for j, col in enumerate(METRIC_COLUMNS):
                    cell_idx = 6 + j
                    v = row[cell_idx]
                    if v == '' or v is None:
                        continue
                    tags: List[str] = []
                    if mark_lowest and row_idx in extremes[col]['min']:
                        tags.append('min')
                    if mark_second and row_idx in extremes[col]['second_max']:
                        tags.append('2nd')
                    if mark_highest and row_idx in extremes[col]['max']:
                        tags.append('max')
                    if tags:
                        row[cell_idx] = f"{v} [" + ','.join(tags) + "]"
            writer.writerow(row)

def output_markdown(records: List[DerRecord], out_file: Optional[Path], extremes: Optional[ExtremesMap] = None,
                    mark_lowest: bool = False, mark_highest: bool = False, mark_second: bool = False):
    # Use HTML table to support grouped headers (colspan) in Markdown renderers.
    base_cols = [
        'model_full', 'model', 'model_type', 'training_dataset', 'eval_set', 'log_path'
    ]
    metric_groups = [
        ('Including no_diac', ['With case', 'Without case']),
        ('Excluding no_diac', ['With case', 'Without case']),
    ]

    def esc(s: str) -> str:
        # Minimal HTML escaping
        return (
            s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
        )

    lines: List[str] = []
    lines.append('<table>')
    lines.append('  <thead>')
    # First header row: base columns (rowspan=2) + group titles (colspan=2)
    lines.append('    <tr>')
    for c in base_cols:
        lines.append(f'      <th rowspan="2">{esc(c)}</th>')
    for title, subs in metric_groups:
        lines.append(f'      <th colspan="{len(subs)}" style="text-align:center">{esc(title)}</th>')
    lines.append('    </tr>')
    # Second header row: subheaders
    lines.append('    <tr>')
    for _, subs in metric_groups:
        for sub in subs:
            lines.append(f'      <th>{esc(sub)}</th>')
    lines.append('    </tr>')
    lines.append('  </thead>')
    lines.append('  <tbody>')
    for row_idx, r in enumerate(records):
        row = r.to_row(COLUMNS)
        lines.append('    <tr>')
        # base columns first
        for v in row[:len(base_cols)]:
            lines.append(f'      <td>{esc(str(v)) if v is not None else ""}</td>')
        # metrics (ensure ordering matches COLUMNS)
        for j, v in enumerate(row[len(base_cols):]):
            # Right align numbers if possible
            try:
                num = float(v) if v != '' else None
            except (TypeError, ValueError):
                num = None
            cell = ''
            attrs = []
            styles = []
            title_tags: List[str] = []
            col_name = METRIC_COLUMNS[j]
            if extremes is not None and (mark_lowest or mark_highest or mark_second):
                if mark_lowest and row_idx in extremes[col_name]['min']:
                    styles.append('background:#d1fae5;font-weight:700')  # greenish
                    title_tags.append('min')
                if mark_second and row_idx in extremes[col_name]['second_max']:
                    styles.append('background:#fef3c7;text-decoration:underline')  # amber underline
                    title_tags.append('2nd')
                if mark_highest and row_idx in extremes[col_name]['max']:
                    styles.append('background:#fee2e2;font-style:italic')  # reddish italic
                    title_tags.append('max')
            if styles:
                attrs.append(f'style="text-align:right;{";".join(styles)}"')
            else:
                attrs.append('style="text-align:right"')
            if title_tags:
                attrs.append(f'title="{esc(", ".join(title_tags))}"')
            if num is None:
                cell = esc(str(v)) if v is not None else ''
            else:
                cell = f'{num:.2f}'
            lines.append(f'      <td {' '.join(attrs)}>{cell}</td>')
        lines.append('    </tr>')
    lines.append('  </tbody>')
    lines.append('</table>')
    content = '\n'.join(lines) + '\n'
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(content, encoding='utf-8')
    else:
        print(content)

def escape_latex(s: str) -> str:
    # Minimal escaping for LaTeX special chars
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    out = ''.join(replacements.get(c, c) for c in s)
    return out

def output_latex(records: List[DerRecord], out_file: Optional[Path], extremes: Optional[ExtremesMap] = None,
                 mark_lowest: bool = False, mark_highest: bool = False, mark_second: bool = False):
    # Build a LaTeX tabular environment with grouped headers
    base_cols = [
        'model_full', 'model', 'model_type', 'training_dataset', 'eval_set', 'log_path'
    ]
    metrics = [
        ('Including no_diac', ['der_w_case_incl_no_diac', 'der_wo_case_incl_no_diac']),
        ('Excluding no_diac', ['der_w_case_excl_no_diac', 'der_wo_case_excl_no_diac']),
    ]

    # Column alignment: left for text, right for numeric metrics
    align_parts: List[str] = []
    for c in base_cols:
        align_parts.append('l')
    for _group, cols in metrics:
        for _ in cols:
            align_parts.append('r')

    lines: List[str] = []
    lines.append('% Auto-generated DER summary table (requires: \\usepackage{booktabs}')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(r'  % Adjust column alignment as needed')
    lines.append('  ' + r'\begin{tabular}{' + ' '.join(align_parts) + '}')
    lines.append('    ' + r'\toprule')
    # Header row 1: base col headers + grouped titles
    hdr1: List[str] = [escape_latex(c) for c in base_cols]
    for group_title, cols in metrics:
        hdr1.append(r'\multicolumn{' + str(len(cols)) + '}{c}{' + escape_latex(group_title) + '}')
    lines.append('    ' + ' & '.join(hdr1) + r' \\')
    # cmidrules under groups
    base_count = len(base_cols)
    cmid_parts: List[str] = []
    cur = base_count
    for _group_title, cols in metrics:
        start = cur + 1
        end = cur + len(cols)
        cmid_parts.append(r'\cmidrule(lr){' + f'{start}-{end}' + '}')
        cur = end
    lines.append('    ' + ' '.join(cmid_parts))
    # Header row 2: subheaders for metrics
    hdr2: List[str] = [''] * base_count
    for _group_title, cols in metrics:
        subs = []
        for c in cols:
            if c.endswith('_w_case_incl_no_diac') or c.endswith('_w_case_excl_no_diac') or 'w_case' in c:
                subs.append('With case')
            elif c.endswith('_wo_case_incl_no_diac') or c.endswith('_wo_case_excl_no_diac') or 'wo_case' in c:
                subs.append('Without case')
            else:
                subs.append(c)
        hdr2.extend([escape_latex(s) for s in subs])
    lines.append('    ' + ' & '.join(hdr2) + r' \\')
    lines.append('    ' + r'\midrule')
    # Rows
    for row_idx, r in enumerate(records):
        row = r.to_row(COLUMNS)
        formatted: List[str] = []
        # base columns
        for v in row[:base_count]:
            formatted.append(escape_latex(v) if v else '')
        # metric columns formatted numeric
        for j, v in enumerate(row[base_count:]):
            if not v:
                formatted.append('')
            else:
                try:
                    num = float(v)
                    s = f'{num:.2f}'
                    if extremes is not None and (mark_lowest or mark_highest or mark_second):
                        col_name = METRIC_COLUMNS[j]
                        # Wrap in LaTeX formatting based on tags (nesting: bold > underline > italic)
                        if mark_lowest and row_idx in extremes[col_name]['min']:
                            s = r'\textbf{' + s + '}'
                        if mark_second and row_idx in extremes[col_name]['second_max']:
                            s = r'\underline{' + s + '}'
                        if mark_highest and row_idx in extremes[col_name]['max']:
                            s = r'\textit{' + s + '}'
                    formatted.append(s)
                except ValueError:
                    formatted.append(escape_latex(v))
        lines.append('    ' + ' & '.join(formatted) + r' \\')
    lines.append('    ' + r'\bottomrule')
    lines.append('  ' + r'\end{tabular}')
    lines.append('  ' + r'\caption{DER summary across models and datasets}')
    lines.append('  ' + r'\label{tab:der_summary}')
    lines.append(r'\end{table}')
    content = '\n'.join(lines) + '\n'
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(content, encoding='utf-8')
    else:
        print(content)

def main():
    parser = argparse.ArgumentParser(description="Collect DER metrics from evaluation logs.")
    parser.add_argument('--results_dir', type=Path, default=Path('results'), help='Root results directory to scan')
    parser.add_argument('--output', type=Path, default=None, help='Optional output file path (CSV or Markdown depending on --format)')
    parser.add_argument('--format', choices=['csv', 'markdown', 'latex'], default='csv', help='Output format')
    parser.add_argument('--mark-lowest', action='store_true', help='Annotate lowest values per metric column')
    parser.add_argument('--mark-highest', action='store_true', help='Annotate highest values per metric column')
    parser.add_argument('--mark-second-highest', action='store_true', help='Annotate second-highest values per metric column')
    args = parser.parse_args()

    logs = list(iter_log_files(args.results_dir))
    records: List[DerRecord] = []
    for log in logs:
        rec = parse_log(log)
        if rec:
            records.append(rec)

    collected = collect(records)

    # Compute extremes only if any marking is requested
    extremes = compute_extremes(collected) if (args.mark_lowest or args.mark_highest or args.mark_second_highest) else None

    if args.format == 'csv':
        output_csv(collected, args.output, extremes, args.mark_lowest, args.mark_highest, args.mark_second_highest)
    elif args.format == 'markdown':
        output_markdown(collected, args.output, extremes, args.mark_lowest, args.mark_highest, args.mark_second_highest)
    else:  # latex
        output_latex(collected, args.output, extremes, args.mark_lowest, args.mark_highest, args.mark_second_highest)

    # Summary to stderr
    import sys
    print(f"Parsed {len(collected)} log(s) from {args.results_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()
