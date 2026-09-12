"""Read-only verification of archived inputs and XLSX results; Python stdlib only."""
import hashlib
import json
import pathlib
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sheets(file):
    with zipfile.ZipFile(file) as z:
        assert z.testzip() is None, file
        names = z.namelist()
        assert not any("externalLinks/" in n or "vbaProject" in n for n in names), file
        strings = []
        if "xl/sharedStrings.xml" in names:
            strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        book = ET.fromstring(z.read("xl/workbook.xml"))
        result = {}
        for sheet in book.find("s:sheets", NS):
            target = rels[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            part = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            cells = {}
            for c in ET.fromstring(z.read(part)).findall(".//s:sheetData/s:row/s:c", NS):
                assert c.attrib.get("t") != "e", (file, sheet.attrib["name"], c.attrib["r"])
                v = c.findtext("s:v", default="", namespaces=NS)
                if c.attrib.get("t") == "s":
                    v = strings[int(v)]
                elif c.attrib.get("t") == "inlineStr":
                    v = "".join(c.find("s:is", NS).itertext())
                cells[c.attrib["r"]] = v
            result[sheet.attrib["name"]] = cells
        return result


def byte_values(text):
    return [int(n.replace(",", "")) for n in re.findall(r"([\d,]+) B(?:/|\s|$)", text)]


def main():
    manifest = json.loads((ROOT / "analysis/snapshot-manifest.json").read_text())
    for item in manifest["files"]:
        actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
        assert actual == item["sha256"], item["path"]
    extension_hashes = 0
    for source_file in (ROOT / 'models').glob('*/sources.json'):
        for item in json.loads(source_file.read_text()).get('artifacts', []):
            assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
            extension_hashes += 1
    workbooks = list((ROOT / "models").glob("*/*.xlsx"))
    for file in workbooks:
        assert list(sheets(file)) == ["Prefill", "Decode"], file
    results = json.loads(subprocess.check_output(["node", str(ROOT / "scripts/export-analysis.mjs"), "communication"], cwd=ROOT))
    expected = json.loads((ROOT / "examples/communication-256k/results.json").read_text())
    assert results == expected, "Regenerated communication results differ from the documented case"
    v41_dir = ROOT / 'models/deepseek-v4.1-flash'
    v41 = json.loads(subprocess.check_output(['node', str(ROOT / 'scripts/export-analysis.mjs'), 'communication', 'deepseek-v4.1-flash'], cwd=ROOT))
    assert v41 == json.loads((v41_dir / 'communication-example.json').read_text())
    for cells in sheets(v41_dir / 'deepseek-v4.1-flash-analysis.xlsx').values():
        assert abs(float(cells['J118']) * 2**30 - 510286023000) < .01
        cached_payload = sum(float(cells[f'P{row}']) for row in (49, 50, 51)) * 2**30
        assert abs(cached_payload - int(v41['cache']['batchLogicalBytesPerReplica'])) < .01
    case = sheets(ROOT / "examples/communication-256k/communication-requirements-20260904.xlsx")
    comparisons = 0
    for result in results:
        cells = case[result["model"]]
        p, d, kv = result["prefill"], result["decode"], result["pdCache"]
        values = {
            "I5": [p["tpBoundary"], d["tpBoundary"]],
            "I7": [p["sp"], d["sp"]],
            "I8": [p["dsaCp"], d["dsaCp"]],
            "I10": [p["attention"]], "I11": [d["attention"]],
            "I12": [kv["total"], kv["attention"], kv["auxiliary"]],
        }
        if int(p["epAllToAll"]):
            values["I16"] = [p["epAllToAll"], d["epAllToAll"]]
            if int(p["sharedExpertTp"]):
                values["I16"] += [p["sharedExpertTp"], d["sharedExpertTp"]]
            else:
                values["I16"].append(0)
        else:
            values["I16"] = [0, 0]
        for cell, expected_bytes in values.items():
            actual = byte_values(cells[cell])
            assert actual == list(map(int, expected_bytes)), (result["model"], cell, actual, expected_bytes)
            comparisons += len(actual)
        for row in range(4, 19):
            for col in ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA"]:
                assert cells.get(f"{col}{row}", "") == "", (result["model"], col, row)
    print(f"Verified {len(manifest['files'])} snapshot hashes, {len(workbooks) + 1} XLSX archives, {comparisons} communication byte values; capability cells remain blank.")
    print(f"Verified {extension_hashes} extension artifact hashes, V4.1 communication reproduction, saved weight total and retained cache bytes.")
    print("This checks saved results and reproduction, not runtime HCCL correctness or Excel recalculation.")


if __name__ == "__main__":
    main()
