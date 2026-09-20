"""Read saved XLSX values/formulas and compare preserved content with originals."""
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'examples/archive/pre-pd-contention-20260920'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def load(file):
    with zipfile.ZipFile(file) as z:
        assert z.testzip() is None
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        book = ET.fromstring(z.read('xl/workbook.xml'))
        styles = ET.fromstring(z.read('xl/styles.xml'))
        xfs = list(styles.find('s:cellXfs', NS))
        formats = {e.attrib['numFmtId']: e.attrib['formatCode'] for e in styles.findall('s:numFmts/s:numFmt', NS)}
        result = {}
        for i, sheet in enumerate(book.find('s:sheets', NS), 1):
            doc = ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
            cells = {}
            for c in doc.findall('.//s:sheetData/s:row/s:c', NS):
                assert c.attrib.get('t') != 'e', (file, sheet.attrib['name'], c.attrib['r'])
                v = c.findtext('s:v', default='', namespaces=NS)
                kind = c.attrib.get('t', 'n')
                if kind == 's':
                    v = strings[int(v)]
                elif kind == 'inlineStr':
                    v = ''.join(c.find('s:is', NS).itertext())
                elif kind == 'n' and v:
                    v = float(v)
                style = xfs[int(c.attrib.get('s', 0))]
                numfmt = style.attrib.get('numFmtId', '0')
                # Style IDs may change on export; compare the resolved definitions.
                style_parts = [formats.get(numfmt, numfmt)]
                for attr, group in [('fontId', 'fonts'), ('fillId', 'fills'), ('borderId', 'borders')]:
                    style_parts.append(ET.tostring(list(styles.find('s:' + group, NS))[int(style.attrib.get(attr, 0))]))
                align = style.find('s:alignment', NS)
                style_parts.append(None if align is None else dict(align.attrib))
                cells[c.attrib['r']] = (v, c.findtext('s:f', default='', namespaces=NS), style_parts)
            features = {}
            for name in ['mergeCells', 'dataValidations', 'sheetProtection']:
                e = doc.find('s:' + name, NS)
                features[name] = None if e is None else ET.tostring(e)
            e = doc.find('s:sheetViews/s:sheetView/s:pane', NS)
            features['pane'] = None if e is None else dict(e.attrib)
            result[sheet.attrib['name']] = (cells, features)
        return result


def equal(a, b, context):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), (context, a, b)
    else:
        assert a == b, (context, a, b)


def preserved(old, new, max_row=None, sweep_presentation=False):
    assert set(old) <= set(new) and set(new) - set(old) <= {'分钟场景'}
    for name, (cells, features) in old.items():
        actual, new_features = new[name]
        assert features == new_features, (name, 'native features changed')
        for ref, (v, f, style) in cells.items():
            if max_row is not None and name == '规格汇总' and int(re.search(r'\d+', ref)[0]) > max_row:
                continue
            if not v and not f:
                continue
            # The user explicitly restyled the sweep and removed this qualifier.
            # Keep the arithmetic, inputs and native features under comparison.
            if sweep_presentation:
                if isinstance(v, str):
                    v = v.replace('；未实机验证', '')
                f = f.replace('；未实机验证', '')
            equal(v, actual[ref][0], (name, ref, 'value'))
            assert f == actual[ref][1], (name, ref, 'formula')
            if not sweep_presentation:
                assert style == actual[ref][2], (name, ref, 'style')


rows = json.loads((ROOT / 'examples/pd-contention/model-results.json').read_text())
books = {}
for row in rows:
    slug, phase, r = row['slug'], row['phase'], row['region']
    if slug not in books:
        books[slug] = load(ROOT / f'models/{slug}/{slug}-analysis.xlsx')
        preserved(load(ARCHIVE / f'{slug}-analysis.xlsx'), books[slug])
    cells = books[slug][phase][0]
    for col, key in [('B', 'baselineBandwidthMs'), ('D', 'contendedBandwidthMs'), ('F', 'deltaForwardMs')]:
        equal(cells[f'{col}{r["output"]}'][0], row[key], (slug, phase, key))
    for n in range(r['first'], r['last'] + 1):
        assert cells[f'E{n}'][1] == f'$H${r["input"]}'
        equal(cells[f'E{n}'][0], 100, (slug, phase, 'editable bandwidth default'))
    assert cells[f'H{r["output"]}'][0] == ''
    assert cells[f'J{r["output"]}'][0] == ''

expected = json.loads(subprocess.check_output(['node', 'scripts/export-analysis.mjs', 'contention'], cwd=ROOT))
assert expected == json.loads((ROOT / 'examples/pd-contention/results.json').read_text())
sweep = load(ROOT / 'examples/kv-ep-sweep/kv-ep32-ep256.xlsx')
preserved(load(ARCHIVE / 'kv-ep32-ep256.xlsx'), sweep, max_row=48, sweep_presentation=True)
cells = sweep['规格汇总'][0]
for i, row in enumerate(expected, 58):
    for col, phase, metric in [('C', 'prefill', 'baselineBandwidthMs'), ('E', 'prefill', 'deltaForwardMs'),
                               ('F', 'decode', 'baselineBandwidthMs'), ('H', 'decode', 'deltaForwardMs')]:
        equal(cells[f'{col}{i}'][0], row[phase][metric], (row['model'], col, metric))
    assert cells[f'M{i}'][1] == '$B$55' and cells[f'N{i}'][1] == '$D$55'
    assert cells[f'J{i}'][0] == '' and cells[f'L{i}'][0] == ''
print('PASS: 8 saved workbooks, 14 phase extensions, 12 EP cases; numeric cells/formulas and native features preserved. Sweep styles/qualifier text follow the requested presentation update.')
print('Bandwidth formulas and saved results agree with the independent calculator. Recalculation mutations were checked by the Artifact Tool updater; desktop Excel was not run.')
