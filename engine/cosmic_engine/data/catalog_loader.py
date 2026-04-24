"""Generic CSV → list-of-dict loader.

Keeps parsing deliberately dumb: values stay as strings here, and any
domain-specific conversion (floats, enums, vector math) happens in the
catalog-type-specific loader (:mod:`cosmic_engine.data.star_catalog`,
etc.).
"""

from __future__ import annotations

import csv
import os


def load_csv(file_path: str) -> list[dict[str, str]]:
    """Read a CSV file and return rows as a list of dicts.

    - Whitespace is stripped from every cell and every header.
    - Rows where every cell is empty after stripping are skipped.
    - All values remain strings; numeric parsing is the caller's job.

    Raises:
        FileNotFoundError: if ``file_path`` does not exist.
        ValueError: if the file is empty or has no header row.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with open(file_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {file_path}") from None

        header = [h.strip() for h in raw_header]
        if not header or all(h == "" for h in header):
            raise ValueError(f"CSV file has no headers: {file_path}")

        rows: list[dict[str, str]] = []
        for raw_row in reader:
            stripped = [cell.strip() for cell in raw_row]
            if not stripped or all(cell == "" for cell in stripped):
                continue
            # Pad short rows and truncate over-long rows to the header width.
            if len(stripped) < len(header):
                stripped = stripped + [""] * (len(header) - len(stripped))
            elif len(stripped) > len(header):
                stripped = stripped[: len(header)]
            rows.append(dict(zip(header, stripped)))
        return rows
