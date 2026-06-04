/** Client-side SMILES extraction for preview/import (mirrors backend parsers). */

const SMILES_COLUMNS = new Set([
  'smiles',
  'smi',
  'canonical_smiles',
  'smiles_string',
])

export function parseSmilesFromTextClient(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean)
}

export async function parseSmilesFromCsvClient(file: File): Promise<string[]> {
  const text = await file.text()
  const lines = text.split(/\r?\n/).filter((l) => l.trim())
  if (!lines.length) return []

  const header = lines[0].split(',').map((c) => c.trim().replace(/^"|"$/g, ''))
  let colIdx = header.findIndex((c) => SMILES_COLUMNS.has(c.toLowerCase()))
  if (colIdx < 0) colIdx = 0

  const out: string[] = []
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',')
    const raw = (parts[colIdx] ?? '').trim().replace(/^"|"$/g, '')
    if (raw && raw.toLowerCase() !== 'nan') out.push(raw)
  }
  return out
}
