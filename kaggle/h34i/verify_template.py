"""Run the real template offline against local mounts; no scored fixtures saved."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT))
import h34i_generality as h


def main():
    checks=[]
    streams=json.loads((ROOT/'data/h34i/manifest.json').read_text())
    template=(BASE/'kernel_template.py').read_text()
    # Keep this separate from verify_offline so numerical smoke remains one process.
    h.h33.configure(1)
    with tempfile.TemporaryDirectory(prefix='h34i-template-') as folder:
        tmp=Path(folder); inp=tmp/'input'; work=tmp/'working'; inp.mkdir(); work.mkdir()
        code=inp/'code'; code.mkdir()
        for name in h.source_hashes():
            (code/name).symlink_to(ROOT/name)
        for stream in h.STREAMS:
            (inp/(stream+'.bin')).symlink_to(ROOT/'data/h34i'/(stream+'.bin'))
        local=template.replace('/kaggle/input',str(inp)).replace('/kaggle/working',str(work)).replace('stop_at=90000','stop_at=144')
        config=dict(arm='dense_6',stream='c4',seed=0,part=1,documents=256,hashes=h.source_hashes(),
                    streams=streams,fingerprint='offline-template',kernel_id='ezzzzzx/h34i-c4-dense-6-s0',parent_id=None)
        script=tmp/'kernel.py'; env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        def execute():
            script.write_text('CONFIG = '+repr(config)+'\n'+local)
            subprocess.run([sys.executable,str(script)],check=True,env=env)
            return json.loads((work/'results.json').read_text())
        first=execute()
        assert (first['offset'],first['pending_bytes'])==(128,16)
        parent=inp/'parent'; work.rename(parent); work.mkdir()
        config.update(part=2,kernel_id='ezzzzzx/h34i-c4-dense-6-s0-part2',parent_id=first['kernel_id'])
        resumed=execute()
        assert resumed['resumed_checkpoint_sha256']==first['checkpoint_sha256']
        raw=h.load_stream('c4'); whole=h.run('dense_6',0,raw[:256])
        for key in ('bits_per_byte','layers','mixer_slices','slices','first_40kb_bits_per_byte'):
            assert whole[key]==resumed[key]
        checks.append(dict(check='real_split_template',n_bytes=256,pending_bytes=16,max_bits_abs=0.))
        # Wrong stream/parent or altered checkpoint must be rejected before learning.
        bad=json.loads((parent/'results.json').read_text()); bad['stream']='bible'
        h.dump(parent/'results.json',bad)
        script.write_text('CONFIG = '+repr(config)+'\n'+local)
        p=subprocess.run([sys.executable,str(script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert p.returncode!=0 and b'AssertionError' in p.stderr
        checks.append(dict(check='wrong_parent_stream_rejected',cases=1))
        # Separate fresh execution covers every stream through the real loader/template.
        for stream in h.STREAMS:
            config.update(part=0,arm='plain_1',stream=stream,parent_id=None,
                          kernel_id=f'ezzzzzx/h34i-{stream}-plain-1-s0')
            scored=execute()
            raw=h.load_stream(stream); whole=h.run('plain_1',0,raw[:256])
            assert scored['bits_per_byte']==whole['bits_per_byte']
            assert scored['raw_sha256']==streams[stream]['sha256'] and scored['stream']==stream
            checks.append(dict(check='real_unsplit_template',stream=stream,n_bytes=256,max_bits_abs=0.))
        # Byte identity is enforced before any training begins.
        (inp/'mnist.bin').unlink(); (inp/'mnist.bin').write_bytes(b'wrong')
        script.write_text('CONFIG = '+repr(config)+'\n'+local)
        p=subprocess.run([sys.executable,str(script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert p.returncode!=0 and b'AssertionError' in p.stderr
        checks.append(dict(check='bad_data_rejected',cases=1))
    h.dump(ROOT/'results/h34i/template-checks.json',dict(ok=True,checks=checks))
    print(json.dumps(dict(ok=True,checks=checks)))


if __name__=='__main__':
    main()
