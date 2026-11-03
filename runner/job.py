"""Durable full-run wrapper: log, status, validation, and resumable failure state."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from .collect import report
from .validate import validate

ROOT=Path('data/master')
CMD=[sys.executable,'-m','runner.collect','--source','flat','--start','2024-10-29','--end','2026-10-28','--output',str(ROOT)]

def status(state,detail):
    Path('STATUS.md').write_text(f'# Collection status\n\n{state}\n\nUpdated {datetime.now(timezone.utc).isoformat()}.\n\n{detail}\n\nLog: `data/master.log`. Resume: `python -m runner.job` using the environment in README.md.\n\nThe full dataset is complete only when this file says COMPLETE and validation has passed. No model has been trained.\n')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'job.lock'
    # OS process checks avoid stale locks preventing resume after restart.
    if lock.exists():
        try:
            pid=int(lock.read_text());__import__('os').kill(pid,0)
        except (ProcessLookupError,ValueError):pass
        else:raise RuntimeError('A collection job is already running')
    lock.write_text(str(__import__('os').getpid()))
    started=time.monotonic()
    try:
        status('RUNNING','Collecting 2024-10-29 through 2026-10-28, plus ten warm-up sessions. Pilot artifacts are in `data/pilot`.')
        with Path('data/master.log').open('a',buffering=1) as log:
            child=subprocess.Popen(CMD,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                time.sleep(30)
                checks=sorted((ROOT/'checkpoints').glob('*.json'))
                audits=[json.loads(p.read_text()) for p in checks]
                target=[a for a in audits if a['target']]
                count=sum(a['eligible'] for a in target)
                last=audits[-1]['day'] if audits else 'none yet'
                status('RUNNING',f'Committed through {last}. Target sessions completed: {len(target)}. Qualifying stock-days: {count}. Elapsed minutes: {(time.monotonic()-started)/60:.1f}. Daily partitions are available immediately; final merged files and validation follow completion.')
            if child.returncode:
                if list((ROOT/'checkpoints').glob('*.json')):report(ROOT)
                status('INTERRUPTED',f'Collector exited with code {child.returncode}. Inspect the final lines of `data/master.log`, fix the cause, then resume the same command. Committed days are preserved.')
                return child.returncode
        validate(ROOT)
        s=json.loads((ROOT/'summary.json').read_text())
        lines=[f"Completed {s['target_days_completed']} target sessions; {s['qualifying_rows']} eligible stock-days across {s['unique_tickers']} tickers. Final Parquet bytes: {s['final_parquet_bytes']:,}.", '', 'Prevalence (positive / negative / unknown):', '']
        lines.extend(f"- {r['horizon']} min, +{r['threshold_pct']}%: {r['positive']} / {r['negative']} / {r['unknown']}" for r in s['prevalence'])
        status('COMPLETE — validation passed','\n'.join(lines))
        return 0
    except Exception:
        status('INTERRUPTED','Wrapper or validation failed. Inspect `data/job.log` and `data/master.log`; committed data is preserved. Resume after fixing the cause.')
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':sys.exit(main())
