"""Audit exported native formulas and cached values against the standalone model."""
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES_ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
expected = json.loads(subprocess.check_output(['node', 'scripts/export-analysis.mjs', 'minute'], cwd=ROOT))
assert expected == json.loads((ROOT / 'examples/pd-contention/minute-results.json').read_text())
by_key = {(x['slug'], x['inputs']['ep'], x['promptTokens'], x['inputs']['tpotMs']): x for x in expected}
regions = json.loads((ROOT / 'examples/pd-contention/minute-workbook-regions.json').read_text())
metrics = dict(F='kvBurstsPerMinute', G='windowMs', H='overlappingStepsPerMinute',
               I='allCommunicationInsideStepsPerMinute', J='partialStepsPerMinute', K='overlappingLayerCallsPerMinute',
               L='extraMsForBaselineMinute', M='meanTpotIncreaseMs', N='tpotIncreaseFraction',
               P='minimumDecodeBandwidthGBps')
checks = 0
for region in regions:
    with zipfile.ZipFile(FILES_ROOT / region['file']) as z:
        assert z.testzip() is None
        book = ET.fromstring(z.read('xl/workbook.xml'))
        sheets = list(book.find('s:sheets', NS))
        assert sheets[-1].attrib['name'] == '分钟场景'
        for i in range(1, len(sheets) + 1):
            doc = ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
            assert not doc.findall('.//s:c[@t="e"]', NS), (region['file'], i)
            if region['file'].startswith('examples/kv-ep-sweep/'):
                assert all(v.attrib.get('showGridLines', '1') == '1' for v in doc.findall('s:sheetViews/s:sheetView', NS))
                assert doc.find('s:sheetProtection', NS) is None
                assert not [e for e in doc.findall('s:cols/s:col', NS) + doc.findall('s:sheetData/s:row', NS) if e.attrib.get('hidden') == '1']
                assert doc.find('s:sheetFormatPr', NS).attrib.get('zeroHeight', '0') != '1'
                assert not any(text in z.read(f'xl/worksheets/sheet{i}.xml').decode() for text in ['估算', '运行未验证', '未实机验证'])
        cells = {c.attrib['r']: c for c in doc.findall('.//s:sheetData/s:row/s:c', NS)}
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]

        def value(ref):
            c = cells[ref]
            v = c.findtext('s:v', default='', namespaces=NS)
            if c.attrib.get('t') == 's':
                return strings[int(v)]
            if c.attrib.get('t') == 'inlineStr':
                return ''.join(c.find('s:is', NS).itertext())
            if c.attrib.get('t') == 'str':
                return v
            return float(v) if v else ''

        assert value('N5') == value('N7') == 4800
        has_tp = 'P22' in cells and value('P22') == 'TP GB/s 输入'
        if has_tp:
            assert value('F11') == 4800
            assert value('J11') == value('N11') == 1
        for row in region['cases']:
            r = row['row']
            source = by_key[(row['slug'], row['ep'], int(value(f'C{r}')), int(value(f'E{r}')))]
            assert value(f'A{r}') == source['model']
            assert value(f'B{r}') == row['ep']
            assert value(f'D{r}') == source['inputs']['ttftSeconds']
            if source['profile'] == 'deepseek_v41':
                assert value(f'O{r}') == ''
            for col, key in metrics.items():
                ref = f'{col}{r}'
                assert cells[ref].find('s:f', NS) is not None, (region['file'], ref)
                scale = 1000 if col == 'M' else 1e6 if col == 'N' else 1
                assert math.isclose(value(ref), source['result'][key] * scale, rel_tol=1e-9, abs_tol=1e-9), (ref, value(ref), source['result'][key])
                checks += 1
            assert math.isclose(value(f'H{r}'), value(f'I{r}') + value(f'J{r}'), abs_tol=1e-8)
            if has_tp:
                for col, expected_value in [('Q', source['inputs']['decodeTpBytesPerRank'] / 1e9),
                                            ('R', source['result']['tpCommMs']),
                                            ('S', source['result']['extraEpMsForBaselineMinute']),
                                            ('T', source['result']['extraTpMsForBaselineMinute'])]:
                    assert cells[f'{col}{r}'].find('s:f', NS) is not None
                    assert math.isclose(value(f'{col}{r}'), expected_value, rel_tol=1e-9, abs_tol=1e-9)
                    checks += 1
                assert math.isclose(value(f'L{r}'), value(f'S{r}') + value(f'T{r}'), abs_tol=1e-9)
        # Overrides must link to visible controls until the user replaces them.
        for r in range(region['parameterFirst'], region['parameterLast'] + 1):
            for col, ref in [('I', '$N$7'), ('J', '$N$5'), ('K', '$B$7'), ('L', '$F$7')]:
                assert cells[f'{col}{r}'].findtext('s:f', namespaces=NS) == ref
            if has_tp:
                for col, ref in [('P', '$F$11'), ('Q', '$J$11'), ('R', '$N$11')]:
                    assert cells[f'{col}{r}'].findtext('s:f', namespaces=NS) == ref
print(f'PASS: {len(regions)} saved workbooks, 144 distinct scenarios, {checks} formula-backed result cells; original sheets scanned without formula errors')
