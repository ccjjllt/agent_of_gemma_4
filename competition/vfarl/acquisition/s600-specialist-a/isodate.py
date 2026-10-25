"""Prepare reviewed ISO parser/calendar contracts for the S600 course.

Reuse the existing fixed public-source cache and acquisition helper. All build
generation is the real upstream setuptools_scm fallback, before snapshots.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];REPLAY=ROOT/'competition/vfarl/replay';ART=REPLAY/'artifacts/s600-specialist-a';BATCH=HERE/'isodate';COMMON=ROOT/'competition/vfarl/acquisition/s600-common-root'
PLANS=[
 ('9dtive5e','S','S1','parse-date','src/isodate/isodates.py','parse_date','ISO_calendar_week_and_ordinal_date_decoding',"from isodate import parse_date\nfrom datetime import date\nr=parse_date('2024-060');print('ordinal_date',r)\nassert r==date(2024,2,29)\nassert parse_date('2024-W09-4')==date(2024,2,29)"),
 ('yli5v1t7','S','S1','parse-time','src/isodate/isotime.py','parse_time','fractional_ISO_time_precision_component_and_zone_decoding',"from isodate import parse_time\nfrom datetime import timedelta\nr=parse_time('12:34:56.123456Z');print('parsed_time',r)\nassert (r.hour,r.minute,r.second,r.microsecond)==(12,34,56,123456)\nassert r.utcoffset()==timedelta(0)"),
 ('xt3b27w3','S','S1','parse-datetime','src/isodate/isodatetime.py','parse_datetime','date_time_designator_split_and_structured_composition',"from isodate import parse_datetime\nr=parse_datetime('2024-02-29T12:34:56Z');print('parsed_datetime',r)\nassert (r.year,r.month,r.day,r.hour,r.minute,r.second)==(2024,2,29,12,34,56)"),
 ('b4wwvhxy','S','S1','parse-duration','src/isodate/isoduration.py','parse_duration','numeric_duration_units_sign_and_result_type_decoding',"from isodate import parse_duration\nfrom datetime import timedelta\nr=parse_duration('P2DT3H4M5S');print('parsed_duration',r)\nassert isinstance(r,timedelta) and r==timedelta(days=2,hours=3,minutes=4,seconds=5)\nassert parse_duration('-P1D')==timedelta(days=-1)"),
 ('ontmajpk','S','S1','date-grammar','src/isodate/isodates.py','build_date_regexps','compiled_ISO_date_grammar_requires_entire_input',"from isodate.isodates import build_date_regexps\np=build_date_regexps();r=any(x.match('2024-02-29') for x in p);print('valid_date_match',r)\nassert r\nassert not any(x.match('2024-02-29junk') for x in p)"),
 ('3o9c1n49','S','S1','time-grammar','src/isodate/isotime.py','build_time_regexps','compiled_ISO_time_pattern_and_capture_protocol',"from isodate.isotime import build_time_regexps\nimport re\np=build_time_regexps();print('pattern_types',[type(x).__name__ for x in p])\nassert all(isinstance(x,re.Pattern) for x in p)\nassert any(x.match('12:34:56Z') for x in p)"),
 ('e11yyls1','S','S1','format-duration','src/isodate/isostrf.py','_strfduration','duration_unit_order_fraction_and_designator_serialization',"from isodate import Duration,strftime\nr=strftime(Duration(years=1,months=2,days=3,hours=4,minutes=5,seconds=6),'%P');print('duration_components',r)\nassert r=='1Y2M3DT4H5M6S'"),
 ('0uycn1hs','S','S1','format-calendar','src/isodate/isostrf.py','_strfdt','calendar_format_field_order_and_year_width',"from isodate import strftime\nfrom datetime import datetime\nr=strftime(datetime(2024,2,29,12,34,56),'%Y-%m-%dT%H:%M:%S');print('calendar_fields',r)\nassert r=='2024-02-29T12:34:56'"),
 ('b9hykg81','S','S4','calendar-add','src/isodate/duration.py','Duration.__add__','month_carry_and_leap_day_clamping_during_calendar_addition',"from isodate import Duration\nfrom datetime import date\nr=date(2024,1,31)+Duration(months=1);print('calendar_addition',r)\nassert r==date(2024,2,29)"),
 ('7wyj1vv4','S','S4','calendar-subtract','src/isodate/duration.py','Duration.__rsub__','reverse_calendar_subtraction_carry_and_month_end_clamping',"from isodate import Duration\nfrom datetime import date\nr=date(2024,3,31)-Duration(months=1);print('calendar_subtraction',r)\nassert r==date(2024,2,29)"),
 ('3ty8lwck','R','R3','duration-subtract','src/isodate/duration.py','Duration.__sub__','duration_subtraction_operator_component_protocol',"from isodate import Duration\nfrom datetime import timedelta\nr=Duration(years=2,months=3,days=4)-Duration(years=1,months=1,days=2);print('duration_difference',r.years,r.months,r.tdelta)\nassert (r.years,r.months,r.tdelta)==(1,2,timedelta(days=2))"),
 ('5fxlcut5','F','F1','date-forwarding','src/isodate/isodates.py','date_isoformat','date_format_wrapper_forwards_value_format_and_year_width',"from isodate import date_isoformat\nfrom datetime import date\nr=date_isoformat(date(2024,2,29));print('date_wrapper',r)\nassert r=='2024-02-29'"),
 ('3roqi4en','F','F1','time-forwarding','src/isodate/isotime.py','time_isoformat','time_format_wrapper_preserves_value_and_format_argument_order',"from isodate import time_isoformat\nfrom datetime import time\nr=time_isoformat(time(12,34,56));print('time_wrapper',r)\nassert r=='12:34:56'"),
 ('6og7s810','F','F1','format-routing','src/isodate/isostrf.py','strftime','duration_or_calendar_value_routes_to_matching_formatter',"from isodate import strftime\nfrom datetime import datetime,timedelta\nr=strftime(datetime(2024,2,29),'%Y-%m-%d');print('routed_calendar',r)\nassert r=='2024-02-29'\nassert strftime(timedelta(days=2),'%P')=='2D'"),
 ('5xq2zv9y','G','G-default_value','datetime-default-format','src/isodate/isodatetime.py','datetime_isoformat','datetime_default_format_preserves_clock_components',"from isodate import datetime_isoformat\nfrom datetime import datetime\nr=datetime_isoformat(datetime(2024,2,29,12,34,56));print('default_datetime',r)\nassert r=='2024-02-29T12:34:56'"),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def label(path):return str(path.relative_to(ROOT))
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def prepare():
    BATCH.mkdir(parents=True,exist_ok=True);(BATCH/'artifacts/pages').mkdir(parents=True,exist_ok=True);(BATCH/'artifacts/refs').mkdir(parents=True,exist_ok=True)
    for page in (COMMON/'artifacts/pages').glob('gweis__isodate*.json'):
        target=BATCH/'artifacts/pages'/page.name
        if not target.exists():target.symlink_to(page.absolute())
    refs=COMMON/'artifacts/refs/swesmith__gweis__isodate.17cb25eb.txt';local_refs=BATCH/'artifacts/refs'/refs.name
    if not local_refs.exists():local_refs.symlink_to(refs.absolute())
    m=load('s600a_iso_public_source',COMMON/'collect.py');m.HERE=BATCH;m.ART=ART/'isodate';m.ART.mkdir(parents=True,exist_ok=True)
    parent_up=REPLAY/'artifacts/s600-common-root/gweis__isodate-upstream';our_up=m.ART/'gweis__isodate-upstream'
    for suffix in ['.tgz','.revision.json']:
        src=parent_up.with_suffix(suffix);dest=our_up.with_suffix(suffix)
        if src.exists() and not dest.exists():dest.symlink_to(src.absolute())
    configs=[]
    for suffix,primary,sub,key,relative,symbol,contract,body in PLANS:
        config={'canonical_repo':'gweis/isodate','snapshot':'17cb25eb','source_layout':'src','task_profile':'.venv-cache-task/lib/python3.12/site-packages','source_license':'BSD-3-Clause','course_membership':'S600_specialist' if primary=='S' else 'S600_common','key':'s600-a-isodate-'+key,'instance_id':'gweis__isodate.17cb25eb.func_basic__'+suffix,'relative_path':relative,'symbol':symbol,'primary_adapter':primary,'submechanism':sub,'family_contract':contract,'contract':contract.replace('_',' '),'body':body}
        m.prepare(config);configs.append(config);print('prepared',config['instance_id'],primary,sub,symbol,flush=True)
    # Source archives lack this generated file. Execute the real upstream
    # configured build generator, with its declared archive fallback version.
    build_records=[]
    for source in [our_up,*[m.ART/(c['key']+'-source') for c in configs]]:
        cfg=source/'pyproject.toml';assert 'fallback_version = "0.0.0.dev0"' in cfg.read_text()
        code="import setuptools_scm,sys;print(setuptools_scm.get_version(root=sys.argv[1],write_to='src/isodate/version.py',fallback_version='0.0.0.dev0'))"
        version=subprocess.check_output([str(REPLAY/'.venv-sfr/bin/python'),'-c',code,str(source)],text=True).strip();assert version=='0.0.0.dev0'
        generated=source/'src/isodate/version.py';build_records.append({'source':label(source),'configuration':label(cfg),'configuration_sha256':sha(cfg),'generator':'setuptools_scm.get_version','actual_version':version,'generated_file':label(generated),'generated_file_sha256':sha(generated)})
    build=BATCH/'source-build-profile.json';build.write_text(json.dumps({'kind':'actual_upstream_setuptools_scm_archive_fallback','write_to':'src/isodate/version.py','fallback_version':'0.0.0.dev0','records':build_records,'no_import_stub_or_core_behavior_replacement':True},indent=2)+'\n')
    tasks=[]
    for c in configs:
        prep=BATCH/(c['key']+'-preparation.json');t=json.loads(prep.read_text());t['source_build_profile']={'path':label(build),'sha256':sha(build)};t['production_symbol']=t['symbol'];t['production_path']=t['relative_path'];tasks.append(t)
    review=HERE/'reviewed-isodate-candidates.json';review.write_text(json.dumps({'source_dataset':'SWE-bench/SWE-smith-py','source_revision':m.REV,'physical_split':'train','scope':'Ten dedicated representation/calendar contracts plus five shared repair contracts, all distinct from roots previously collected isodate functions. Publisher descriptions, all278 original nodes and six full source test files retained. Real upstream build fallback generated before snapshots. Prepared candidates are not admitted successes.','tasks':tasks},ensure_ascii=False,indent=2)+'\n');print(label(review),sha(review))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'));parser.add_argument('--start',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--attempt',type=int,default=1);a=parser.parse_args()
    if a.stage=='prepare':prepare()
    else:load('s600a_iso_run',HERE/'collect.py').run(a.start,a.limit,'isodate',a.attempt)
