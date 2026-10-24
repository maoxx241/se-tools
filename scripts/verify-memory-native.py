"""Change one public-example input and verify recalculation in headless LibreOffice.

Only a disposable temporary copy is changed. Original workbooks are read-only.
Requires LibreOffice, Python standard library, and Node.js.
"""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent.parent
READER = runpy.run_path(str(ROOT / 'scripts/read-memory-xlsx.py'))
NS = READER['NS']
read_xlsx = READER['read']


def verify(source, soffice, node, description=None):
    original = read_xlsx(source)
    old = original['sheets'][0]['cells']['R2']['value']
    changed = old / 2
    assert changed >= 1 and changed.is_integer()
    script = """
import {loadMemoryModel,loadMemoryDescription} from './lib/model-memory/catalog.mjs';
import {memoryReport} from './scripts/build-memory-workbook.mjs';
const model = process.argv[2] ? await loadMemoryDescription(process.argv[2]) : await loadMemoryModel('kimi-k3');
model.scenarios[0].preset.inputs.tp = Number(process.argv[1]);
console.log(JSON.stringify(memoryReport(model)));
"""
    args = [node, '--input-type=module', '-e', script, str(changed)]
    if description:
        args.append(str(description.resolve()))
    expected = json.loads(subprocess.check_output(args, cwd=ROOT))
    with tempfile.TemporaryDirectory(prefix='model-memory-native-') as folder:
        temporary = Path(folder)
        input_dir, output_dir = temporary / 'input', temporary / 'output'
        input_dir.mkdir()
        output_dir.mkdir()
        probe = input_dir / 'probe.xlsx'
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(probe, 'w', zipfile.ZIP_DEFLATED) as dst:
            for entry in src.infolist():
                data = src.read(entry.filename)
                if entry.filename.startswith('xl/worksheets/') and entry.filename.endswith('.xml'):
                    xml = ET.fromstring(data)
                    for cell in xml.findall('.//m:sheetData/m:row/m:c', NS):
                        if entry.filename == 'xl/worksheets/sheet1.xml' and cell.attrib['r'] == 'R2':
                            cell.find('m:v', NS).text = str(int(changed))
                        if cell.find('m:f', NS) is not None:
                            # Force the external application to compute formulas,
                            # rather than validate an authoring engine's old cache.
                            cached = cell.find('m:v', NS)
                            if cached is not None:
                                cell.remove(cached)
                    data = ET.tostring(xml, encoding='utf-8', xml_declaration=True)
                dst.writestr(entry, data)
        command = [soffice, '--headless', f'-env:UserInstallation={(temporary / "profile").as_uri()}',
                   '--convert-to', 'xlsx', '--outdir', str(output_dir), str(probe)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        result_file = output_dir / 'probe.xlsx'
        if run.returncode or not result_file.exists():
            raise RuntimeError(f'LibreOffice conversion failed: {run.stdout} {run.stderr}')
        result = read_xlsx(result_file)
        assert [s['name'] for s in result['sheets']] == [s['name'] for s in expected['scenarios']]
        for sheet, scenario in zip(result['sheets'], expected['scenarios']):
            cells = sheet['cells']
            for address, key in [('R23', 'weightGiB'), ('R25', 'bufferGiB'), ('R27', 'cacheGiB'), ('R28', 'totalGiB')]:
                actual, wanted = cells[address]['value'], scenario['stages'][0][key]
                assert isinstance(actual, (int, float)) and abs(actual - wanted) < 1e-9 * max(1, abs(wanted)), (sheet['name'], address, actual, wanted)
                assert cells[address]['formula']
            for i, row in enumerate(scenario['stages'][0]['rows'], 2):
                actual = cells[f'K{i}']['value']
                assert abs(actual - row['bytes'] / 2**30) < 1e-9 * max(1, abs(actual)), (sheet['name'], i)
        assert result['sheets'][0]['cells']['R2']['value'] == changed
    return {'model': expected['model'], 'engine': 'LibreOffice', 'nativeRecalculation': 'passed',
            'changedFirstSheetTP': [old, changed], 'sourceWorkbook': 'unchanged'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--soffice', default='soffice')
    parser.add_argument('--node', default='node')
    parser.add_argument('--spec', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.file, args.soffice, args.node, args.spec), indent=2))
