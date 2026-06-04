"""Parse SMILES from text, TXT, or CSV content."""

from __future__ import annotations

import io
from typing import BinaryIO

import pandas as pd

SMILES_COLUMN_NAMES = frozenset(
    {"smiles", "smi", "canonical_smiles", "smiles_string"}
)


def detect_smiles_column(columns: list[str]) -> str | None:
    for col in columns:
        if str(col).lower() in SMILES_COLUMN_NAMES:
            return col
    return None


def parse_smiles_from_text(text: str) -> list[str]:
    """One SMILES per line from plain text."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_smiles_from_csv_bytes(data: bytes) -> list[str]:
    """Extract SMILES column from CSV bytes (utf-8, fallback gbk)."""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8", errors="replace")

    smiles_col = None
    for col in df.columns:
        if str(col).lower() in SMILES_COLUMN_NAMES:
            smiles_col = col
            break
    if smiles_col is None and len(df.columns) > 0:
        smiles_col = df.columns[0]

    if smiles_col is None:
        return []

    smiles_list = df[smiles_col].astype(str).tolist()
    return [
        s.strip()
        for s in smiles_list
        if s and str(s).lower() != "nan" and str(s).strip()
    ]


def parse_smiles_from_file(
    file_obj: BinaryIO, filename: str, *, text_content: str | None = None
) -> list[str]:
    """Parse SMILES from uploaded file based on extension."""
    name = filename.lower()
    if name.endswith(".csv"):
        data = file_obj.read()
        return parse_smiles_from_csv_bytes(data)
    if name.endswith(".txt"):
        raw = file_obj.read()
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        return parse_smiles_from_text(text)
    if text_content is not None:
        return parse_smiles_from_text(text_content)
    raise ValueError("Unsupported file type. Use .txt or .csv")
