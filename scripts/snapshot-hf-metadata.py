"""Snapshot public HF config and grouped Safetensors headers, never tensor data.

Example: python3 scripts/snapshot-hf-metadata.py --repo deepseek-ai/DeepSeek-V4.1-Flash
  --revision <40-character commit> --output outputs/v41-metadata
Python standard library only. Output must be a new directory.
"""
import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import re
import struct
import urllib.parse
import urllib.request


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, start=None, end=None):
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        if start is not None:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                raise ValueError("Server did not honor byte range; refusing to download weights")
            data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError("Unexpected range length")
        else:
            data = response.read(32 * 1024 * 1024 + 1)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("Metadata exceeds 32 MiB limit")
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', args.repo) or not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('Provide owner/repository and an immutable 40-character revision')
    args.output.mkdir(parents=True, exist_ok=False)
    base = f'https://huggingface.co/{args.repo}/resolve/{args.revision}/'
    config = fetch(base + 'config.json')
    index_bytes = fetch(base + 'model.safetensors.index.json')
    index = json.loads(index_bytes)
    shards = sorted(set(index['weight_map'].values()))

    def header(shard):
        url = base + urllib.parse.quote(shard, safe='')
        length = struct.unpack('<Q', fetch(url + '?header=length', 0, 7))[0]
        if not 0 < length < 20_000_000:
            raise ValueError('Invalid Safetensors header length')
        raw = fetch(url + '?header=body', 8, length + 7)
        return shard, raw, json.loads(raw)

    counts, seen, headers, total = collections.Counter(), {}, [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for shard, raw, tensors in pool.map(header, shards):
            headers.append({'file': shard, 'headerBytes': len(raw), 'sha256': digest(raw)})
            for name, tensor in tensors.items():
                if name == '__metadata__':
                    continue
                if name in seen or index['weight_map'].get(name) != shard:
                    raise ValueError('Header/index mismatch or duplicate tensor')
                seen[name] = shard
                size = tensor['data_offsets'][1] - tensor['data_offsets'][0]
                if size < 0:
                    raise ValueError('Invalid tensor offsets')
                key = (re.sub(r'\.\d+\.', '.N.', name), tensor['dtype'], tuple(tensor['shape']), size)
                counts[key] += 1
                total += size
    if seen != index['weight_map'] or total != index['metadata']['total_size']:
        raise ValueError('Tensor coverage/total size does not match index')
    snapshot = {
        'repository': args.repo, 'revision': args.revision,
        'collected': datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        'method': 'HTTP Range: 8-byte length and JSON header only; no tensor payload downloaded',
        'tensorCount': len(seen), 'totalPayloadBytes': total,
        'indexSha256': digest(index_bytes), 'configSha256': digest(config), 'headers': headers,
        'groups': [{'name': name, 'dtype': dtype, 'storedShape': shape, 'bytesPerTensor': size, 'count': count}
                   for (name, dtype, shape, size), count in sorted(counts.items())],
    }
    (args.output / 'config.json').write_bytes(config)
    (args.output / 'weight-metadata.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(shards)} headers, {len(seen)} tensors, {total} payload bytes described; no tensor data downloaded')


if __name__ == '__main__':
    main()
