"""Fixed headline-only proxies. No learned sentiment or outcome-informed taxonomy."""
import re
import numpy as np
import pandas as pd

FINANCING=re.compile(r'\b(public offering|registered direct|private placement|share offering|stock offering|offering of .{0,50}shares|at.the.market|warrants?|convertible notes?|dilution|prospectus)\b',re.I)
EARNINGS=re.compile(r'\b(earnings|quarterly results|financial results|reports? .{0,15}(quarter|revenue|eps)|guidance|outlook|forecast)\b',re.I)
CONTRACT=re.compile(r'\b(wins?|awarded|secures?|signs?|receives?|announces?)\b.{0,70}\b(contract|order|agreement|partnership|deal)\b',re.I)
REGULATORY=re.compile(r'\b(fda|regulatory approval|clinical trial|phase [123i]|primary endpoint|trial results)\b',re.I)
POSITIVE=re.compile(r'\b(raises?|increases?|lifts?)\b.{0,30}\b(guidance|outlook|forecast)|\b(beats?|tops?|exceeds?)\b.{0,30}\b(estimates?|expectations?|consensus)|\b(wins?|awarded|secures?)\b.{0,50}\b(contract|order)|\b(fda|regulator)\b.{0,25}\b(approves?|clears?)|\b(receives?|obtains?)\b.{0,20}\bfda\b.{0,15}\b(approval|clearance)|\b(meets?|met)\b.{0,20}\bprimary endpoint',re.I)
NEGATIVE=re.compile(r'\b(misses?|cuts?|lowers?|fails?|rejects?|denies|withdraws?|reduces?)\b',re.I)
HYPOTHETICAL=re.compile(r'\b(may|might|could|seeks?|awaits?|pending|ahead of|preview|expected to)\b',re.I)
STOPWORDS={'inc','incorporated','corp','corporation','ltd','limited','plc','co','company','holdings','holding','group','class','common','stock','shares','the','sa','nv'}


def company_specific(article,ticker,name):
    tickers=article.get('tickers',[])
    if ticker not in tickers:return False
    if len(tickers)==1:return True
    title=article.get('title','')
    if re.search(r'(?:\$|NASDAQ\s*:\s*|NYSE\s*:\s*|AMEX\s*:\s*)'+re.escape(ticker)+r'\b',title,re.I):return True
    words=[w for w in re.findall(r'[a-z0-9]+',name.lower()) if w not in STOPWORDS]
    title_words=' '.join(re.findall(r'[a-z0-9]+',title.lower()))
    if len(words)>=2:return ' '.join(words[:2]) in title_words
    return len(words)==1 and len(words[0])>=5 and words[0] in title_words.split()


def classify_title(title):
    flags=dict(financing=bool(FINANCING.search(title)),earnings_guidance=bool(EARNINGS.search(title)),
               contract=bool(CONTRACT.search(title)),regulatory_trial=bool(REGULATORY.search(title)))
    flags['positive']=bool(POSITIVE.search(title) and not NEGATIVE.search(title) and not HYPOTHETICAL.search(title) and not flags['financing'])
    return flags


def news_features(articles,ticker,name,cutoff,prior_close):
    eligible=[];audit=[];seen=set()
    for article in articles:
        stamp=pd.to_datetime(article.get('published_utc'),utc=True,errors='coerce')
        if pd.isna(stamp) or stamp>=cutoff:continue
        # If an update timestamp is supplied, disallow known later revisions.
        updated=pd.to_datetime(article.get('last_updated_utc') or article.get('updated_utc'),utc=True,errors='coerce')
        if not pd.isna(updated) and updated>=cutoff:continue
        title=article.get('title','');normalized=' '.join(re.findall(r'[a-z0-9]+',title.lower()))
        if normalized in seen:continue
        seen.add(normalized)
        specific=company_specific(article,ticker,name);fresh=stamp>=prior_close
        flags=classify_title(title)
        audit.append(dict(id=article.get('id'),title=title,published_utc=stamp.isoformat(),url=article.get('article_url'),specific=specific,fresh=fresh,**flags))
        if specific:eligible.append((stamp,fresh,flags))
    result={'news_articles_relevant':len(eligible),'news_fresh_relevant':sum(fresh for _,fresh,_ in eligible)}
    fresh=[(stamp,flags) for stamp,is_fresh,flags in eligible if is_fresh]
    for flag in ('financing','earnings_guidance','contract','regulatory_trial','positive'):
        result['news_'+flag+'_count']=sum(flags[flag] for _,flags in fresh)
    result['news_minutes_since_fresh']=float((cutoff-max(s for s,_ in fresh)).total_seconds()/60) if fresh else np.nan
    result['news_classified_fresh']=sum(any(flags[k] for k in ('financing','earnings_guidance','contract','regulatory_trial')) for _,flags in fresh)
    result['news_positive_candidate']=int(result['news_positive_count']>0 and result['news_financing_count']==0)
    result['news_has_financing']=int(result['news_financing_count']>0)
    result['news_none_classified']=int(result['news_classified_fresh']==0)
    return result,audit
