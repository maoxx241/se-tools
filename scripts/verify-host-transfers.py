"""Read-only verification of the published default two-sheet host-transfer XLSX."""
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples/host-transfers/engram-h2d-prefix-rh2d.xlsx"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
report = json.loads((FILE.parent / "results.json").read_text())

with zipfile.ZipFile(FILE) as z:
    assert z.testzip() is None
    book = ET.fromstring(z.read("xl/workbook.xml"))
    sheets = book.findall("s:sheets/s:sheet", NS)
    assert [s.attrib["name"] for s in sheets] == ["Engram H2D", "Prefix RH2D"]
    assert not book.findall("s:workbookProtection", NS)
    for s in sheets:
        assert s.attrib.get("state", "visible") == "visible"
    parsed = []
    for n in (1, 2):
        sheet = ET.fromstring(z.read(f"xl/worksheets/sheet{n}.xml"))
        assert not sheet.findall("s:sheetProtection", NS)
        for v in sheet.findall("s:sheetViews/s:sheetView", NS):
            assert v.attrib.get("showGridLines", "1") in ("1", "true")
        for row in sheet.findall("s:sheetData/s:row", NS):
            assert row.attrib.get("hidden", "0") not in ("1", "true")
        for col in sheet.findall("s:cols/s:col", NS):
            assert col.attrib.get("hidden", "0") not in ("1", "true")
        cells = {c.attrib["r"]: c for c in sheet.findall("s:sheetData/s:row/s:c", NS)}
        assert not [a for a, c in cells.items() if c.attrib.get("t") == "e"]
        assert sheet.findall("s:dataValidations/s:dataValidation", NS)
        parsed.append(cells)

    def number(cells, cell):
        return float(cells[cell].find("s:v", NS).text)

    def near(cells, cell, expected, scale=1):
        actual = number(cells, cell) * scale
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-6), (cell, actual, expected)

    e, p = parsed
    for r, item in enumerate(report["engram"], 12):
        for col, key, scale in [("G", "vectorBytesPerRank", 2**20), ("H", "auxiliaryBytesPerRank", 2**20),
                                ("I", "totalBytesPerRank", 2**20), ("J", "totalBytesPerNode", 2**30),
                                ("K", "totalBytesPerEP", 2**30), ("L", "leaderBytes", 2**20)]:
            near(e, f"{col}{r}", item[key], scale)
            assert e[f"{col}{r}"].find("s:f", NS) is not None
    for r, item in enumerate(report["prefix"], 12):
        for col, key, scale in [("G", "cacheBytesPerRank", 2**20), ("H", "stateBytesPerRank", 2**20),
                                ("I", "requestBytesAllTP", 2**20), ("J", "rankBatchBytes", 2**30),
                                ("K", "dpBatchBytes", 2**30), ("L", "epBatchBytes", 2**30)]:
            near(p, f"{col}{r}", item[key], scale)
            assert p[f"{col}{r}"].find("s:f", NS) is not None
    # Independent scalar checks of the accounting conventions, not file shape alone.
    near(e, "L8", 2 * 3 * 8 * 256 * 2)
    near(e, "G12", 48)
    near(e, "G13", 3)
    near(e, "I12", 48 + 1.5 + 144 / 2**20)
    near(p, "I42", (40 * 128 * 1024 + (3 * 8192 + 16384) * 1154) * 8 / 2**20)
    near(p, "G36", 23 * 2 * 256 * 2 * 16384 / 2**20)
    for base in range(12, 48, 6):
        for offset in range(3):
            near(p, f"L{base + offset + 3}", number(p, f"L{base + offset}") * 8)
            near(p, f"J{base + offset + 3}", number(p, f"J{base + offset}"))

print("PASS: saved XLSX has two editable visible sheets, native formulas, and all 46 default cases match; independent byte/TP/EP checks pass.")
