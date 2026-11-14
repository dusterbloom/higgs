#!/usr/bin/env python3
"""Compare identical radix and retained Clover turns on a fresh benchmark server.

Use the same --case-key after restarting each comparison binary. Reusing a
populated radix cache invalidates a fresh-conversation comparison. TTFT includes
request processing through the first streamed text; it is not pure GPU prefill.
"""
import argparse
import copy
import json
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--prefix-words', type=int, default=2048)
parser.add_argument('--turns', type=int, default=6)
parser.add_argument('--case-key', default='ttft-reference-v1')
args = parser.parse_args()
base=args.base_url.rstrip('/'); sid=int(time.time()*1000)
rows=[];nonce=args.case_key

def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as r:return json.load(r)

def request(label,messages,session=None):
 d=metrics()
 if d['capacity']['activeReservations']:raise RuntimeError('Other inference active; refusing contaminated measurement')
 before=d['totals']['requests']
 body={'model':'clover','messages':messages,'temperature':0,'max_tokens':4,'enable_thinking':False,'speculation':'self_spec','stream':True,'return_progress':True,'stream_options':{'include_usage':True}}
 if session is not None:body['session_id']=session
 t=time.perf_counter();first=None;text='';usage={};progress=[]
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   x=json.loads(raw)
   if 'prompt_progress' in x:progress.append({'at_s':round(time.perf_counter()-t,4),'data':x['prompt_progress']})
   if x.get('usage'):usage=x['usage']
   for choice in x.get('choices',[]):
    v=choice.get('delta',{}).get('content') or ''
    if v:
     if first is None:first=time.perf_counter()-t
     text+=v
 total=time.perf_counter()-t
 after=metrics()
 delta={k:after['cache'][k]-d['cache'][k] for k in ('continuations','radix_hits','session_prompt_prefix_misses')}
 if after['totals']['requests']-before != 1:
  raise RuntimeError('Other requests contaminated measurement')
 if label.startswith('retained-') and label != 'retained-seed':
  if delta['continuations'] != 1 or delta['session_prompt_prefix_misses'] != 0:
   raise RuntimeError('Retained route fell back instead of continuing: '+str(delta))
  previous=next(r for r in reversed(rows) if r['session'])
  cached=usage.get('prompt_tokens_details',{}).get('cached_tokens')
  if cached != previous['usage']['prompt_tokens']:
   raise RuntimeError('Unexpected retained checkpoint boundary: '+str(cached))
 row={'label':label,'route_delta':delta,'session':session is not None,'ttft_s':first,'total_s':total,'usage':usage,'progress':progress,'other_requests':after['totals']['requests']-before-1,'text':text,'cache':{k:after['cache'][k] for k in ('continuations','radix_hits','session_prompt_prefix_misses')}}
 rows.append(row);print(json.dumps(row),flush=True)
 with open(args.output,'w') as out:json.dump(rows,out,indent=2)
 return text

messages=[{'role':'system','content':'Cache timing experiment '+nonce+'. Reply only OK. Reference: '+(' alpha'*args.prefix_words)},{'role':'user','content':'Acknowledge.'}]
request('radix-seed',messages)
seed='OK'
# Establish retained state from the identical prompt. This seed can hit radix;
# neither seed is included in follow-up medians or treated as a cold comparison.
request('retained-seed',messages,sid)
for turn in range(args.turns):
 messages=copy.deepcopy(messages)+[{'role':'assistant','content':seed},{'role':'user','content':('additional reference detail. '*32)+f'Turn {turn}. Acknowledge.'}]
 request('radix-'+str(turn),messages)
 request('retained-'+str(turn),messages,sid)
