"""Draft F5 contracts part3: click, records, mido (16)."""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
C = []

def add(key, iid, canon, snap, rel, layout, module, symbol, family, contract, body):
    assert "assert" in body and ("==" in body), key
    C.append({"canonical_repo": canon, "snapshot": snap, "relative_path": rel,
              "source_layout": layout, "task_profile": ".venv/lib/python3.12/site-packages",
              "source_license": "UPSTREAM",
              "key": key, "instance_id": iid, "symbol": symbol,
              "primary_adapter": "F", "submechanism": "F5",
              "family_contract": family, "contract": contract, "body": body})

# ---------- click 8 ----------
add("click-with-resource", "pallets__click.fde47b4b.func_basic__86655824",
    "pallets/click", "fde47b4b", "src/click/core.py", "src", "click.core", "Context.with_resource",
    "with_resource_enters_once_and_defers_exit_to_close",
    "Registering a resource must enter it exactly once via the exit stack and defer its exit to close; entering manually plus via the stack double-enters across the produce-consume-close chain.",
"""import click
from click import Context, Command
enters=[]; exits=[]; closed=[]
class R:
    def __enter__(self):
        enters.append('enter')
        return 'RES'
    def __exit__(self,*a):
        exits.append('exit')
        return False
ctx=Context(Command('x'))
r=ctx.with_resource(R())
ctx.call_on_close(lambda: closed.append('cb'))
first=[list(enters),r]
ctx.close()
print('resource_enter_sequence',[first,list(exits),list(closed)])
assert [first,list(exits),list(closed)]==[[['enter'],'RES'],['exit'],['cb']]""")

add("click-close-intelligently", "pallets__click.fde47b4b.func_basic__05bca298",
    "pallets/click", "fde47b4b", "src/click/utils.py", "src", "click.utils", "LazyFile.close_intelligently",
    "close_intelligently_closes_only_owned_files",
    "Intelligent close must close files the wrapper owns and leave foreign handles open; inverting the flag leaks owned files and closes foreign ones across the open-use-close chain.",
"""import tempfile, os
from click.utils import LazyFile
fd1, p1=tempfile.mkstemp(); os.close(fd1)
fd2, p2=tempfile.mkstemp(); os.close(fd2)
try:
    f=LazyFile(p1,'w'); f.open(); f.write('x'); f.should_close=True; f.close_intelligently()
    owned_closed=f._f.closed
    g=LazyFile(p2,'w'); g.open(); g.write('y'); g.should_close=False; g.close_intelligently()
    unowned_open=(not g._f.closed)
    g.close()
    h=LazyFile(p1,'w')
    with h as hh:
        hh.open().write('z')
    with_closed=h._f.closed
    print('close_intelligently_sequence',[owned_closed,unowned_open,with_closed])
    assert [owned_closed,unowned_open,with_closed]==[True,True,True]
finally:
    for p in (p1,p2):
        try: os.unlink(p)
        except OSError: pass""")

add("click-call-on-close", "pallets__click.fde47b4b.func_basic__87fee394",
    "pallets/click", "fde47b4b", "src/click/core.py", "src", "click.core", "Context.call_on_close",
    "call_on_close_defers_until_close",
    "Teardown callbacks must not run at registration but in LIFO order at close; executing immediately breaks the register-close-fire chain.",
"""from click import Context, Command
ctx=Context(Command('x'))
seen=[]
ctx.call_on_close(lambda: seen.append('cb1'))
ctx.call_on_close(lambda: seen.append('cb2'))
deferred=list(seen)
ctx.close()
print('deferred_close_sequence',[deferred,list(seen)])
assert [deferred,list(seen)]==[[],['cb2','cb1']]
ctx2=Context(Command('y'))
seen2=[]
ctx2.call_on_close(lambda: seen2.append('only'))
ctx2.close()
assert seen2==['only']""")

add("click-context-enter", "pallets__click.fde47b4b.func_basic__788be484",
    "pallets/click", "fde47b4b", "src/click/core.py", "src", "click.core", "Context.__enter__",
    "context_enter_pushes_and_returns_owner",
    "Entering a context must bump depth, push onto the stack, and return itself so nested bodies run inside it; decrementing and returning None breaks the enter-invoke-exit chain.",
"""from click import Context, Command
ctx=Context(Command('x'))
d0=ctx._depth
res=ctx.__enter__()
mid=[res is ctx, ctx._depth==d0+1]
ctx.__exit__(None,None,None)
print('context_enter_sequence',[mid[0],mid[1],ctx._depth==d0])
assert [mid[0],mid[1],ctx._depth==d0]==[True,True,True]""")

add("click-context-close", "pallets__click.fde47b4b.func_basic__89b37a06",
    "pallets/click", "fde47b4b", "src/click/core.py", "src", "click.core", "Context.close",
    "context_close_drains_old_stack_then_renews",
    "Closing must drain the registered callbacks/resources before renewing the stack so teardown fires once and reuse starts clean; renewing first leaks the old teardown across the register-close-reuse chain.",
"""from click import Context, Command
ctx=Context(Command('x'))
order=[]
ctx.call_on_close(lambda: order.append('first'))
ctx.call_on_close(lambda: order.append('second'))
exits=[]
class R:
    def __enter__(self): return 'R'
    def __exit__(self,*a): exits.append('rexit'); return False
ctx.with_resource(R())
ctx.close()
first_close=[list(order),list(exits)]
order2=[]
ctx.call_on_close(lambda: order2.append('third'))
ctx.close()
print('context_close_sequence',[first_close,list(order2)])
assert [first_close,list(order2)]==[[['second','first'],['rexit']],['third']]""")

add("click-lazy-enter", "pallets__click.fde47b4b.func_basic__06363946",
    "pallets/click", "fde47b4b", "src/click/utils.py", "src", "click.utils", "LazyFile.__enter__",
    "lazy_enter_returns_owner_without_opening",
    "Entering a lazy file must return the wrapper without opening the underlying file; eagerly opening and returning None breaks the enter-write-close chain.",
"""import tempfile, os
fd, path=tempfile.mkstemp(); os.close(fd)
try:
    from click.utils import LazyFile
    lf=LazyFile(path,'w')
    res=lf.__enter__()
    checks=[res is lf, lf._f is None]
    lf.open().write('abc')
    lf.__exit__(None,None,None)
    checks.append(open(path).read()=='abc')
    print('lazy_enter_sequence',checks)
    assert checks==[True,True,True]
finally:
    try: os.unlink(path)
    except OSError: pass""")

add("click-progress-enter", "pallets__click.fde47b4b.func_basic__cda1646e",
    "pallets/click", "fde47b4b", "src/click/_termui_impl.py", "src", "click._termui_impl", "ProgressBar.__enter__",
    "progress_enter_arms_and_returns_bar_for_iteration",
    "Entering a progress bar must arm the entered flag and return the bar so iteration yields every item; leaving it disarmed and returning None breaks the start-iterate-finish chain.",
"""from io import StringIO
from click._termui_impl import ProgressBar
bar=ProgressBar([1,2,3], length=3, file=StringIO())
res=bar.__enter__()
checks=[res is bar, bar.entered is True]
consumed=list(bar)
bar.__exit__(None,None,None)
print('progress_enter_sequence',[checks[0],checks[1],consumed])
assert [checks[0],checks[1],consumed]==[True,True,[1,2,3]]""")

add("click-keepopen-enter", "pallets__click.fde47b4b.func_basic__0b319698",
    "pallets/click", "fde47b4b", "src/click/utils.py", "src", "click.utils", "KeepOpenFile.__enter__",
    "keepopen_enter_returns_owner_and_never_closes",
    "Entering a keep-open wrapper must return the wrapper and its exit must never close the underlying file; returning None breaks the enter-use-exit chain.",
"""import tempfile, os
fd, path=tempfile.mkstemp(); os.close(fd)
try:
    from click.utils import KeepOpenFile
    f=open(path,'w')
    kf=KeepOpenFile(f)
    first=(kf.__enter__() is kf)
    kf.__exit__(None,None,None)
    still_open=(not f.closed)
    with kf as k2:
        second=(k2 is kf)
    f.close()
    print('keepopen_enter_sequence',[first,still_open,second])
    assert [first,still_open,second]==[True,True,True]
finally:
    try: os.unlink(path)
    except OSError: pass""")

# ---------- records 3 ----------
add("records-conn-enter", "kennethreitz__records.5941ab27.func_basic__duvnvyd3",
    "kennethreitz/records", "5941ab27", "records.py", ".", "records", "Connection.__enter__",
    "connection_enter_returns_live_connection",
    "Entering a pooled connection must return the live connection so queries run inside it before exit closes; returning None breaks the acquire-query-close chain.",
"""import records
db=records.Database('sqlite:///:memory:')
conn=db.get_connection()
enter_ok=(conn.__enter__() is conn)
conn.__exit__(None,None,None)
closed_ok=(conn.open is False)
rows=db.query('select 1 as v', fetchall=True).all()
print('conn_enter_sequence',[enter_ok,closed_ok,rows[0]['v']])
assert [enter_ok,closed_ok,rows[0]['v']]==[True,True,1]""")

add("records-txn-commit", "kennethreitz__records.5941ab27.func_basic__fankxhc0",
    "kennethreitz/records", "5941ab27", "records.py", ".", "records", "Database.transaction",
    "transaction_commits_on_success_rolls_back_on_error",
    "A transaction context must commit inserted rows on clean exit and roll back on error before closing; rolling back on success loses the begin-write-commit chain.",
"""import records
db=records.Database('sqlite:///:memory:')
db.bulk_query('create table t (id integer primary key, v text)')
with db.transaction() as conn:
    conn.bulk_query(\"insert into t (v) values ('a')\")
after_commit=[r['v'] for r in db.query('select v from t', fetchall=True).all()]
try:
    with db.transaction() as conn:
        conn.bulk_query(\"insert into t (v) values ('b')\")
        raise ValueError('boom')
except ValueError:
    pass
after_rollback=[r['v'] for r in db.query('select v from t', fetchall=True).all()]
print('txn_commit_sequence',[after_commit,after_rollback])
assert [after_commit,after_rollback]==[['a'],['a']]""")

add("records-txn-begin", "kennethreitz__records.5941ab27.func_basic__gxa7v1n3",
    "kennethreitz/records", "5941ab27", "records.py", ".", "records", "Connection.transaction",
    "connection_transaction_returns_committable",
    "Opening a transaction must return an object with commit and rollback so the begin-write-end chain can settle; returning None or a raw start breaks settling.",
"""import records
db=records.Database('sqlite:///:memory:')
db.bulk_query('create table t (id integer primary key, v text)')
conn=db.get_connection()
tx=conn.transaction()
has_settle=hasattr(tx,'commit') and hasattr(tx,'rollback')
conn.close()
with db.transaction() as c2:
    c2.bulk_query(\"insert into t (v) values ('a')\")
rows=[r['v'] for r in db.query('select v from t', fetchall=True).all()]
print('txn_begin_sequence',[has_settle,rows])
assert [has_settle,rows]==[True,['a']]""")

# ---------- mido 5 ----------
add("mido-file-header", "mido__mido.a0158ff9.func_pm_ctrl_invert_if__u6l8t3wt",
    "mido/mido", "a0158ff9", "mido/midifiles/midifiles.py", ".", "mido.midifiles.midifiles", "read_file_header",
    "file_header_rejects_non_mthd_then_parses_header",
    "Reading a MIDI file must reject non-MThd chunks with OSError before parsing the six header bytes; swapping the branches misroutes valid and invalid files across the chunk-header-track chain.",
"""import io, struct
from mido.midifiles.midifiles import read_file_header
valid=io.BytesIO(b'MThd'+struct.pack('>L',6)+struct.pack('>hhh',1,1,480))
parsed=read_file_header(valid)
try:
    read_file_header(io.BytesIO(b'XXXX'+struct.pack('>L',6)+b'123456'))
    invalid='parsed'
except OSError:
    invalid='raised'
try:
    read_file_header(io.BytesIO(b'MThd'+struct.pack('>L',6)+b'ab'))
    short='parsed'
except EOFError:
    short='eof'
print('file_header_sequence',[tuple(parsed),invalid,short])
assert [tuple(parsed),invalid,short]==[(1,1,480),'raised','eof']""")

add("mido-chunk-header", "mido__mido.a0158ff9.func_basic__pkrfwgpd",
    "mido/mido", "a0158ff9", "mido/midifiles/midifiles.py", ".", "mido.midifiles.midifiles", "read_chunk_header",
    "chunk_header_requires_eight_bytes_big_endian",
    "Chunk headers must be exactly eight big-endian bytes; accepting short reads and decoding little-endian corrupts every later size across the chunk-header-track chain.",
"""import io, struct
from mido.midifiles.midifiles import read_chunk_header
full=read_chunk_header(io.BytesIO(struct.pack('>4sL',b'MTrk',100)))
try:
    read_chunk_header(io.BytesIO(b'short'))
    short='parsed'
except EOFError:
    short='eof'
print('chunk_header_sequence',[(full[0],full[1]),short])
assert [(full[0],full[1]),short]==[(b'MTrk',100),'eof']""")

add("mido-header-unpack", "mido__mido.a0158ff9.func_basic__re7p62np",
    "mido/mido", "a0158ff9", "mido/midifiles/midifiles.py", ".", "mido.midifiles.midifiles", "read_file_header",
    "file_header_unpacks_name_size_order_exact_length",
    "Header parsing must unpack name-then-size, read exactly size bytes, and require six data bytes; swapping fields and reading size-plus-one misaligns the header-track-end chain.",
"""import io, struct
from mido.midifiles.midifiles import read_file_header
valid=io.BytesIO(b'MThd'+struct.pack('>L',6)+struct.pack('>hhh',1,2,480))
parsed=read_file_header(valid)
try:
    read_file_header(io.BytesIO(b'XXXX'+struct.pack('>L',6)+b'123456'))
    invalid='parsed'
except OSError:
    invalid='raised'
try:
    read_file_header(io.BytesIO(b'MThd'+struct.pack('>L',6)+b'ab'))
    short='parsed'
except EOFError:
    short='eof'
print('header_unpack_sequence',[tuple(parsed),invalid,short])
assert [tuple(parsed),invalid,short]==[(1,2,480),'raised','eof']""")

add("mido-tick2second", "mido__mido.a0158ff9.func_basic__ffdh8rzb",
    "mido/mido", "a0158ff9", "mido/midifiles/units.py", ".", "mido.midifiles.units", "tick2second",
    "tick_to_second_scales_by_tempo_over_division",
    "Tick conversion must scale by tempo microseconds over division with no offset; inverting the fraction and adding a second breaks the tick-second-tick chain.",
"""from mido.midifiles.units import tick2second
values=[tick2second(t,480,500000) for t in (0,480,960)]
print('tick_second_sequence',values)
assert values==[0.0,0.5,1.0]""")

add("mido-fix-eot", "mido__mido.a0158ff9.func_basic__mavs1eg3",
    "mido/mido", "a0158ff9", "mido/midifiles/tracks.py", ".", "mido.midifiles.tracks", "fix_end_of_track",
    "fix_end_of_track_preserves_order_and_accumulates_forward",
    "End-of-track fixing must scan forward, accumulate only end markers, add deltas forward, and close once; scanning backward with subtraction and off-by-one close scrambles the message-timing-end chain.",
"""from mido import Message, MetaMessage
from mido.midifiles.tracks import fix_end_of_track
msgs=[Message('note_on',note=60,velocity=64,time=10),Message('note_on',note=61,velocity=64,time=20),MetaMessage('end_of_track',time=30)]
fixed=[(m.type,m.time) for m in fix_end_of_track(msgs)]
print('eot_fix_sequence',fixed)
assert fixed==[('note_on',10),('note_on',20),('end_of_track',30)]""")

(HERE / "contracts-part3.json").write_text(json.dumps({"date": "2026-10-25", "tasks": C}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote part3 {len(C)}")
