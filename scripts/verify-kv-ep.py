"""Dependency-free audit of the saved 10-case JSON and Excel cached results."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'examples/kv-ep-sweep'
expected = json.loads(subprocess.check_output(['node', 'scripts/export-analysis.mjs', 'kv-ep'], cwd=ROOT))
assert expected == json.loads((CASE / 'results.json').read_text())
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(CASE / 'kv-ep32-ep256.xlsx') as z:
    sheets = ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', ns)
    assert [s.attrib['name'] for s in sheets] == ['规格汇总', 'KV明细', 'EP通信']
    for name in ('sheet1.xml', 'sheet2.xml', 'sheet3.xml'):
        doc = ET.fromstring(z.read('xl/worksheets/' + name))
        assert not [c.attrib['r'] for c in doc.findall('.//s:c', ns) if c.attrib.get('t') == 'e'], name
    doc = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    cells = {c.attrib['r']: c for c in doc.findall('.//s:c', ns)}
    for i, r in enumerate(expected):
        n, m = 22 + i, 6 + i
        checks = {f'G{n}': r['cache']['generatedBytes'], f'H{n}': r['cache']['pullBytes'],
                  f'I{n}': r['batch']['generatedBytes'], f'J{n}': r['batch']['pullBytes'],
                  f'K{n}': r['wholePrefillGroup']['generatedBytes'], f'L{n}': r['wholePrefillGroup']['pullBytes'],
                  f'I{m}': r['prefillEP']['modeledBytes'], f'J{m}': r['decodeEP']['modeledRemoteWriteBytes']}
        for ref, byte_count in checks.items():
            assert cells[ref].find('s:f', ns) is not None, f'{ref}: missing formula'
            actual = float(cells[ref].find('s:v', ns).text) * 2**30
            assert abs(actual - int(byte_count)) < 1, (ref, actual, byte_count)
        for ref, tokens in [(f'G{m}', r['stepWorkload']['prefillTokensPerEPGroup']),
                            (f'H{m}', r['stepWorkload']['decodeTokensPerEPGroup'])]:
            assert int(cells[ref].find('s:v', ns).text) == int(tokens)
print('PASS: 10 JSON cases, 80 saved formula byte values, 20 DP-workload values, three sheets without formula errors')
