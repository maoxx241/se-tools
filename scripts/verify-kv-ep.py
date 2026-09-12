"""Dependency-free audit of saved JSON and Excel, including qualified V4.1 plans."""
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
def transfer_estimate(row):
    return row['pullBytes'] if row['pullBytes'] is not None else row['plannedPullBytes']
with zipfile.ZipFile(CASE / 'kv-ep32-ep256.xlsx') as z:
    sheets = ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', ns)
    assert [s.attrib['name'] for s in sheets] == ['规格汇总', 'KV明细', 'EP通信']
    for name in ('sheet1.xml', 'sheet2.xml', 'sheet3.xml'):
        doc = ET.fromstring(z.read('xl/worksheets/' + name))
        assert not [c.attrib['r'] for c in doc.findall('.//s:c', ns) if c.attrib.get('t') == 'e'], name
    doc = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    cells = {c.attrib['r']: c for c in doc.findall('.//s:c', ns)}
    for i, r in enumerate(expected):
        n, m = 24 + i, 6 + i
        checks = {f'G{n}': r['cache']['generatedBytes'], f'H{n}': transfer_estimate(r['cache']),
                  f'I{n}': r['batch']['generatedBytes'], f'J{n}': transfer_estimate(r['batch']),
                  f'K{n}': r['wholePrefillGroup']['generatedBytes'], f'L{n}': transfer_estimate(r['wholePrefillGroup']),
                  f'I{m}': r['prefillEP']['modeledBytes'], f'J{m}': r['decodeEP']['modeledRemoteWriteBytes']}
        for ref, byte_count in checks.items():
            assert cells[ref].find('s:f', ns) is not None, f'{ref}: missing formula'
            actual = float(cells[ref].find('s:v', ns).text) * 2**30
            assert abs(actual - int(byte_count)) < 1, (ref, actual, byte_count)
        for ref, tokens in [(f'G{m}', r['stepWorkload']['prefillTokensPerEPGroup']),
                            (f'H{m}', r['stepWorkload']['decodeTokensPerEPGroup'])]:
            assert int(cells[ref].find('s:v', ns).text) == int(tokens)
        if r['profile'] == 'deepseek_v41':
            assert r['cache']['pullBytes'] is None
            assert '规划值' in cells[f'M{n}'].find('s:f', ns).text
print(f'PASS: {len(expected)} JSON cases, {len(expected)*8} saved formula byte values, '
      f'{len(expected)*2} DP-workload values, qualified V4.1 plans, three sheets without formula errors')
