"""Offline tooling checks; no learner runs, credentials, or network calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import dispatch
import summarize


def main():
    checks = 0
    assert dispatch.MAX_RUNNING == 5
    checks += 1
    assert dispatch.DATASET == 'ezzzzzx/pdp-code-w10'
    checks += 1
    for report, code, expected in [('ready', 0, 'ready'), ('403 Client Error', 1, 'missing'),
                                    ('processing', 0, 'processing'), ('failed', 0, 'failed')]:
        assert dispatch.dataset_kind(report, code) == expected
        checks += 1
    assert dispatch.within_window(None, dispatch.datetime.now(dispatch.timezone.utc))
    checks += 1
    try:
        dispatch.check_layout([], {})
    except RuntimeError as exc:
        assert '12 preregistered' in str(exc)
    else:
        raise AssertionError('accepted empty manifest')
    checks += 1
    with tempfile.TemporaryDirectory(prefix='w10-tooling-', dir='/tmp') as folder:
        out = Path(folder)
        (out / 'smoke.json').write_text(json.dumps(dict(ok=True, workers=1, wall_s=0)))
        with contextlib.redirect_stdout(io.StringIO()):
            summarize.summarize(out, pending=True)
        result = json.loads((out / 'results.json').read_text())
        assert result['pending'] and len(result['missing']) == 48 and not result['runs']
        checks += 1
        assert all(v['n'] == 0 and v['mean'] is None for v in result['gaps'].values())
        checks += 1
        assert all(v['n'] == 0 and v['mean'] is None for arms in result['arm_metrics'].values() for v in arms.values())
        checks += 1
        try:
            summarize.summarize(out)
        except (AssertionError, FileNotFoundError):
            pass
        else:
            raise AssertionError('strict assembly accepted missing scored lives')
        checks += 1
    print(f'offline tooling checks: {checks}/{checks}')


if __name__ == '__main__':
    main()
