"""Check mount discovery, checkpoint chaining and provenance without training.

The numerical loop is exercised by h34j_long.py --smoke. Here only run() is
stubbed; the generated template's real loaders and provenance checks execute.
"""
import json
import shutil
from pathlib import Path
import tempfile
from unittest.mock import patch
import dispatch as d
h=d.h


def main():
    manifest,fingerprint=d.prepare()
    template=(d.BASE/'kernel_template.py').read_text()
    with tempfile.TemporaryDirectory(dir=d.OUT,prefix='template-') as temp:
        root=Path(temp); inputs=root/'input'; inputs.mkdir()
        (inputs/'code').mkdir()
        for name in manifest['hashes']:
            shutil.copyfile(d.ROOT/name,inputs/'code'/name)
        (inputs/'enwik8').symlink_to(d.ROOT/'data/enwik8/enwik8')
        previous=None; calls=[]
        def run(arm,seed,raw,**kwargs):
            assert len(raw)==h.N_DATA and arm=='dense_8' and seed==1
            assert kwargs['stop_at'] in (256,512,h.N_DATA)
            if previous is None: assert 'resume' not in kwargs
            else: assert h.sha(Path(kwargs['resume']).read_bytes())==h.sha((previous/'checkpoint.pt').read_bytes())
            Path(kwargs['checkpoint']).write_bytes(str(kwargs['stop_at']).encode())
            calls.append(kwargs['stop_at'])
            return dict(arm=arm,seed=seed,n_bytes=h.N_DATA,first_40kb_bits_per_byte=8.)
        for part,(start,end) in enumerate(((0,256),(256,512),(512,h.N_DATA)),1):
            working=root/f'working{part}'; working.mkdir()
            if previous:
                shutil.copytree(previous,inputs/'parent')
            tag=f'dense_8-s1-p{part}'
            cfg=dict(hashes=manifest['hashes'],arm='dense_8',seed=1,part=part,start=start,end=end,
                fingerprint=fingerprint,raw_sha256=manifest['raw_sha256'],sha256_e10m=manifest['sha256_e10m'],
                kernel_id='ezzzzzx/h34j-'+tag.replace('_','-'),
                parent_id=f'ezzzzzx/h34j-dense-8-s1-p{part-1}' if part>1 else None)
            code=template.replace('/kaggle/input',str(inputs)).replace('/kaggle/working',str(working))
            with patch.object(h,'run',run):
                exec(compile(code,'<h34j-template>','exec'),{'CONFIG':cfg})
            row=json.loads((working/'results.json').read_text())
            assert row['first_200kb_bits_per_byte']==8. and 'first_40kb_bits_per_byte' not in row
            if part>1:
                assert row['resumed_checkpoint_sha256']==h.sha((previous/'checkpoint.pt').read_bytes())
                # Reject a wrong parent identity before invoking run().
                bad=dict(cfg,parent_id='ezzzzzx/wrong')
                with patch.object(h,'run',side_effect=AssertionError('must not run')):
                    try: exec(compile(code,'<wrong-parent>','exec'),{'CONFIG':bad})
                    except AssertionError as exc: assert str(exc)!='must not run'
                    else: raise AssertionError('Wrong parent accepted')
            if (inputs/'parent').exists(): shutil.rmtree(inputs/'parent')
            previous=working
        assert calls==[256,512,h.N_DATA]
        bad=dict(cfg,raw_sha256='0'*64)
        with patch.object(h,'run',side_effect=AssertionError('must not run')):
            try: exec(compile(code,'<wrong-data>','exec'),{'CONFIG':bad})
            except AssertionError as exc: assert str(exc)!='must not run'
            else: raise AssertionError('Wrong data accepted')
    report=dict(ok=True,code_fingerprint=fingerprint,parts_chained=3,wrong_parents_rejected=2,wrong_stream_rejected=1,numerical_loop_stubbed=True)
    d.dump(d.OUT/'template-checks.json',report); print(json.dumps(report))


if __name__=='__main__': main()
