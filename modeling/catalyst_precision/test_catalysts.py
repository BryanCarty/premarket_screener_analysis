import pandas as pd
from modeling.catalyst_precision.news import news_features,classify_title,company_specific
from modeling.barrier_study.build import barrier_label


def article(title,time='2026-10-28T11:00:00Z',**extra):
    return dict(id=title,title=title,published_utc=time,tickers=['TEST'],**extra)


def test_fixed_positive_and_financing_rules():
    assert classify_title('Test raises full-year revenue guidance')['positive']
    assert classify_title('Test wins a new government contract')['positive']
    assert not classify_title('Test could win a new government contract')['positive']
    assert not classify_title('Test cuts guidance despite revenue growth')['positive']
    assert classify_title('Test announces registered direct offering')['financing']


def test_news_cutoff_update_dedup_and_sentiment_exclusion():
    cutoff=pd.Timestamp('2026-10-28T13:45:00Z');prior=pd.Timestamp('2026-10-27T20:00:00Z')
    a=article('Test raises guidance')
    future=article('Test wins new contract','2026-10-28T13:46:00Z')
    revised=article('Test wins government contract',updated_utc='2026-10-28T14:00:00Z')
    f,audit=news_features([a,a,future,revised],'TEST','Test Inc',cutoff,prior)
    assert f['news_fresh_relevant']==1 and f['news_positive_candidate']==1
    a['insights']=[{'sentiment':'negative'}];a['description']='Future stock collapse'
    assert news_features([a],'TEST','Test Inc',cutoff,prior)[0]==f
    assert len(audit)==1


def test_multi_ticker_market_story_is_not_specific_and_stale_is_not_fresh():
    a=article('Markets rally');a['tickers']=['TEST','OTHER']
    assert not company_specific(a,'TEST','Test Technologies Inc')
    a['title']='Test Technologies wins contract'
    assert company_specific(a,'TEST','Test Technologies Inc')
    f,_=news_features([article('Test raises guidance','2026-10-27T15:00:00Z')],'TEST','Test Inc',pd.Timestamp('2026-10-28T13:45Z'),pd.Timestamp('2026-10-27T20:00Z'))
    assert f['news_positive_candidate']==0


def test_delayed_entry_ignores_earlier_barriers_and_times_out_correctly():
    b=pd.DataFrame(dict(minute=range(585,675),open=100.,high=100.5,low=99.5,close=100.,volume=1000.))
    b.loc[b.minute.eq(600),'high']=150
    b.loc[b.minute.eq(664),'close']=100.2
    r=barrier_label(b,.03,start=605,end=665)
    assert r['outcome']=='neither' and r['exit_minute']==665
