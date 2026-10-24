"""Check published memory XLSX formulas/caches against fresh dependency-free JS results.

This checks saved artifacts. It does not execute Excel formulas; use the JS
verifier or LibreOffice for input-driven recalculation.
"""
import argparse
import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parent.parent
read_xlsx = runpy.run_path(str(ROOT / 'scripts/read-memory-xlsx.py'))['read']


def close(actual, expected, label):
    assert isinstance(actual, (int, float)) and abs(actual - expected) <= 1e-9 * max(1, abs(expected)), f'{label}: {actual} != {expected}'


def column(n):
    result = ''
    while n:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def verify(slug, output_root, node, spec=None):
    directory = output_root / slug
    command = [node, str(ROOT / 'scripts/export-analysis.mjs'), 'memory']
    command.extend(['--spec', str(spec.resolve())] if spec else [slug])
    fresh = json.loads(subprocess.check_output(command, cwd=ROOT))[0]
    saved = json.loads((directory / 'memory-results.json').read_text())
    assert saved == fresh, f'{slug}: saved numeric report is stale'
    book = read_xlsx(directory / f'{slug}-memory.xlsx')
    assert [s['name'] for s in book['sheets']] == [s['name'] for s in fresh['scenarios']]
    total_formulas = 0
    for sheet, scenario in zip(book['sheets'], fresh['scenarios']):
        cells = sheet['cells']
        val = lambda addr: cells[addr]['value']
        formula = lambda addr: cells[addr]['formula']
        pp = len(scenario['stages'])
        first_stage = max(86, len(scenario['stages'][0]['rows']) + 4) + 1
        maximum = max(92, first_stage + pp + 1)
        assert val('U1') == 'Made by mindie-agent' and 'U1:V2' in sheet['merges']
        assert sheet['pane'] and sheet['pane'].get('xSplit') == '2' and sheet['pane'].get('ySplit') == '1', f'{sheet["name"]}: frozen pane missing'
        assert 'R22' in sheet['validations'] and 'R13' in sheet['validations'] and 'R14' in sheet['validations']
        assert formula('R10') is None, 'Shared expert TP must remain an independent default'
        for stage in scenario['stages']:
            r = first_stage + stage['stage']
            for c, key in [('W', 'weightGiB'), ('X', 'cacheGiB'), ('Y', 'totalGiB'), ('AC', 'bufferGiB')]:
                close(val(f'{c}{r}'), stage[key], f'{sheet["name"]}/{c}{r}')
                assert formula(f'{c}{r}')
            for i, tensor in enumerate(stage['rows'], 2):
                assert f'{val(f"A{i}")}/{val(f"B{i}")}' == tensor['name']
                assert val(f'D{i}') == tensor['dtype']
                close(val(f'J{i}'), tensor['globalBytes'] / 2**30, f'{sheet["name"]}/J{i}')
                address = f'{column(27 + stage["stage"])}{i}'
                close(val(address), tensor['bytes'] / 2**30, f'{sheet["name"]}/{address}')
                assert formula(address) and formula(f'J{i}') and formula(f'K{i}')
                if stage['stage'] == 0:
                    close(val(f'K{i}'), tensor['bytes'] / 2**30, f'{sheet["name"]}/K{i}')
                    assert json.loads(val(f'N{i}')) == tensor['shape']
                    close(val(f'O{i}'), tensor['repeat'], f'{sheet["name"]}/O{i}')
        initial = scenario['stages'][0]
        for address, key in [('R23', 'weightGiB'), ('R25', 'bufferGiB'), ('R27', 'cacheGiB'), ('R28', 'totalGiB')]:
            close(val(address), initial[key], f'{sheet["name"]}/{address}')
            assert formula(address)
        close(val(f'Y{maximum}'), scenario['maxTotalGiB'], f'{sheet["name"]}/maximum')
        total_formulas += sum(bool(c['formula']) for c in cells.values())
    return {'model': slug, 'sheets': len(book['sheets']), 'savedFormulas': total_formulas,
            'savedValues': 'match fresh JS calculations', 'nativeRecalculation': 'not exercised by this check'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--model', choices=['kimi-k3'])
    parser.add_argument('--node', default='node')
    args = parser.parse_args()
    targets = [(args.model or 'kimi-k3', args.root or ROOT / 'models', None)]
    if not args.root and not args.model:
        targets.append(('synthetic-hybrid', ROOT / 'examples/model-memory', ROOT / 'examples/model-memory/synthetic.json'))
    print(json.dumps([verify(slug, directory, args.node, spec) for slug, directory, spec in targets], indent=2))
