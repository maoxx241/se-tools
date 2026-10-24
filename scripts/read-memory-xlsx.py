"""Read saved XLSX cells/formulas/validation with only Python's standard library."""
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import PurePosixPath

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
CELL = re.compile(r'(?<![A-Za-z0-9_])(\$?)([A-Z]{1,3})(\$?)([1-9][0-9]*)(?![A-Za-z0-9_(])')


def coordinates(address):
    m = re.fullmatch(r'([A-Z]+)(\d+)', address)
    col = 0
    for c in m[1]:
        col = col * 26 + ord(c) - 64
    return col, int(m[2])


def column(n):
    value = ''
    while n:
        n, remainder = divmod(n - 1, 26)
        value = chr(65 + remainder) + value
    return value


def translate(formula, origin, destination):
    a, b = coordinates(origin), coordinates(destination)

    def shift(match):
        col, row = coordinates(match[2] + match[4])
        col += 0 if match[1] else b[0] - a[0]
        row += 0 if match[3] else b[1] - a[1]
        if col < 1 or row < 1:
            raise ValueError('Shared formula shifted outside sheet')
        return match[1] + column(col) + match[3] + str(row)

    # References inside Excel string literals must not be translated.
    parts = re.split(r'("(?:[^"]|"")*")', formula)
    return ''.join(part if i % 2 else CELL.sub(shift, part) for i, part in enumerate(parts))


def read(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip():
            raise ValueError('Invalid ZIP member CRC')
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings = [''.join(si.itertext()) for si in root]
        rels = {n.attrib['Id']: n.attrib['Target'] for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        book = ET.fromstring(z.read('xl/workbook.xml'))
        sheets = []
        for entry in book.findall('m:sheets/m:sheet', NS):
            target = rels[entry.attrib[REL]]
            target = target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl') / target)
            root = ET.fromstring(z.read(target))
            cells, shared = {}, {}
            for c in root.findall('.//m:sheetData/m:row/m:c', NS):
                f = c.find('m:f', NS)
                if f is not None and f.attrib.get('t') == 'shared' and f.text:
                    shared[f.attrib['si']] = (c.attrib['r'], f.text)
            for c in root.findall('.//m:sheetData/m:row/m:c', NS):
                address = c.attrib['r']
                f, v = c.find('m:f', NS), c.find('m:v', NS)
                dtype = c.attrib.get('t', 'n')
                text = v.text if v is not None else None
                if dtype == 's':
                    value = strings[int(text)] if text is not None else None
                elif dtype == 'inlineStr':
                    value = ''.join(c.find('m:is', NS).itertext())
                elif dtype in ('str', 'e'):
                    value = text
                elif dtype == 'b':
                    value = text == '1'
                else:
                    value = float(text) if text is not None else None
                formula = None
                if f is not None:
                    if f.attrib.get('t') == 'shared':
                        origin, original = shared[f.attrib['si']]
                        formula = translate(original, origin, address)
                    else:
                        formula = f.text
                    if not formula:
                        raise ValueError(f'Missing formula in {entry.attrib["name"]}!{address}')
                    formula = '=' + formula
                if dtype == 'e':
                    raise ValueError(f'Formula error in {entry.attrib["name"]}!{address}: {value}')
                if value is not None or formula:
                    cells[address] = {'value': value, 'formula': formula}
            sheets.append({
                'name': entry.attrib['name'], 'cells': cells,
                'merges': [n.attrib['ref'] for n in root.findall('m:mergeCells/m:mergeCell', NS)],
                'validations': [n.attrib['sqref'] for n in root.findall('m:dataValidations/m:dataValidation', NS)],
                'pane': next((n.attrib for n in root.findall('m:sheetViews/m:sheetView/m:pane', NS)), None),
            })
        return {'sheets': sheets}


if __name__ == '__main__':
    print(json.dumps(read(sys.argv[1]), ensure_ascii=False, separators=(',', ':')))
