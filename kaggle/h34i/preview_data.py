"""Print the exact smoke data, with literal 28-by-28 hexadecimal MNIST rows."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    smoke=json.loads((ROOT/'results/h34i/smoke.json').read_text())
    lines=[]
    for stream in ('c4','bible','mnist'):
        row=smoke['streams'][stream]
        raw=(ROOT/'data/h34i'/row['file']).read_bytes()
        assert len(raw)==row['n_bytes']==180000
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        lines.append(f"{stream} bytes={len(raw)} sha256={row['sha256']}")
        if stream=='mnist':
            lines.extend(' '.join(f'{v:02x}' for v in raw[i:i+28]) for i in range(0,784,28))
        else:
            lines.append(''.join(chr(v) if 32<=v<=126 else '.' for v in raw[:200]))
        scores=[r for r in smoke['log'] if r['check']=='new_stream_plain_1' and r['stream']==stream]
        assert {r['seed'] for r in scores}=={0,1}
        for score in scores:
            lines.append(f"plain_1 seed={score['seed']} bytes=8192 bits={score['bits_per_byte']:.12f}")
        lines.append('')
    text='\n'.join(lines)
    (ROOT/'results/h34i/data-preview.txt').write_text(text)
    print(text)


if __name__=='__main__':
    main()
