import pandas as pd
import exchange_calendars as xcals
from runner.filter_prior_day import count_day


class FakeS3:
    def __init__(self,path): self.path=path
    def head_object(self,**kwargs): return {'ContentLength':self.path.stat().st_size,'ETag':'fixture'}


def test_counts_exact_grid_excludes_extended_hours_and_rejects_duplicates(tmp_path):
    day='2025-07-03'
    opening=pd.Timestamp(day+' 09:30',tz='America/New_York')
    closing=opening+pd.Timedelta(minutes=3)
    raw=tmp_path/(day+'.csv.gz')
    # A complete prior session for A; B has a gap despite duplicate rows.
    records=[]
    for t,offsets in [('A',[-1,0,1,2,3]),('B',[0,0,2])]:
        records.extend({'ticker':t,'window_start':(opening+pd.Timedelta(minutes=x)).value} for x in offsets)
    pd.DataFrame(records).to_csv(raw,index=False,compression='gzip')
    result=count_day(day,{'A','B'},opening,closing,tmp_path,FakeS3(raw))
    assert result['expected_minutes']==3
    assert result['counts']=={'A':3,'B':2}
    assert result['duplicates']['B']==1
    assert not raw.exists()
    # Resumption uses the completed compact audit, without another download.
    assert count_day(day,{'A','B'},opening,closing,tmp_path,None)==result


def test_previous_session_handles_holiday_weekend_and_early_close():
    cal=xcals.get_calendar('XNYS')
    previous=cal.previous_session(pd.Timestamp('2025-07-07'))
    assert str(previous.date())=='2025-07-03'
    assert int((cal.session_close(previous)-cal.session_open(previous)).total_seconds()/60)==210
