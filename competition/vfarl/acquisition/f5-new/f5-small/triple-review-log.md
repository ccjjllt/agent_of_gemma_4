# F5-small triple review log (2026-10-25)

Scope: Smith-py train `77cab905…13558e`, F5 event/stream/lifecycle timing only. Same 5 snapshots as f5-replace (trio cfbbe2c1, freezegun 5f171db0, channels a144b4b8, daphne 32ac73e1, tornado d5ac65c1), all outside 75 banned (f5-replace pilots 0 eligible, no extra ban). Four competition repos excluded.
Filter: single-file patch + F2P single file + F2P+P2P_in_file<50 (full F2P+P2P<50 yields 0 candidates for these 5 repos: min full totals daphne 63, channels 70, freezegun 128, trio 684, tornado 1157; single-file-overall yields 0; per-file small yields 113 -> 68 distinct canon::sym -> 40 selected). 17 BAD excluded read-only (6 dups +10 single-point +1 multi, overlapping): hq96g33n/2l53f5xe/auc2kn82/7ml3iktm/q8tz7b2d/gte8bpt6 dups; l3r7e4bf/gvbsi1f0/u1sosk2w/i7x3of2o/i50xxyek/d6l5ckls/ls0ee38t/d7uuw2nv/62ckl2tv/doej1tg5 single-point; j5yd2544 multi.
Each item: patch92856 single-function confinement + live-on-3.12 + temporal sequence (produce->consume->end) + single-point/escape rejection. 38 strong-ordering +2 neutral, 0 weak in final 40; all 40 distinct canon::sym.

## KEEP (40) — temporal chain asserted in recipe run_command

- KEEP `lm_rewrite__rl4yuxyq` tornadoweb/tornado::task_done tornado/queues.py — F2P1+P2P_in2=3 (full P2P1156), hits ['expired', 'get_nowait']
- KEEP `func_basic__dl7smbl0` python-trio/trio::rate src/trio/_core/_mock_clock.py — F2P1+P2P_in4=5 (full P2P729), hits ['MockClock']
- KEEP `func_basic__ly27fik4` python-trio/trio::put_nowait src/trio/_core/_unbounded_queue.py — F2P1+P2P_in4=5 (full P2P729), hits ['put_nowait', 'UnboundedQueue']
- KEEP `lm_rewrite__fcfzyjzg` python-trio/trio::? src/trio/_core/_mock_clock.py — F2P1+P2P_in4=5 (full P2P729), hits ['MockClock']
- KEEP `func_basic__5ly62yyy` python-trio/trio::empty src/trio/_core/_unbounded_queue.py — F2P1+P2P_in4=5 (full P2P729), hits ['UnboundedQueue']
- KEEP `func_basic__5c2wl96f` python-trio/trio::get_batch_nowait src/trio/_core/_unbounded_queue.py — F2P4+P2P_in1=5 (full P2P726), hits ['UnboundedQueue']
- KEEP `func_basic__h9e4lzy3` python-trio/trio::deadline_to_sleep_time src/trio/_core/_mock_clock.py — F2P1+P2P_in4=5 (full P2P729), hits ['deadline', 'MockClock']
- KEEP `lm_rewrite__fk4sug43` python-trio/trio::jump src/trio/_core/_mock_clock.py — F2P1+P2P_in4=5 (full P2P729), hits ['deadline', 'MockClock']
- KEEP `lm_rewrite__6y3hq768` python-trio/trio::autojump_threshold src/trio/_core/_mock_clock.py — F2P1+P2P_in4=5 (full P2P729), hits ['MockClock']
- KEEP `combine_file__3c4joxmg` python-trio/trio::__anext__ src/trio/_core/_unbounded_queue.py — F2P4+P2P_in1=5 (full P2P726), hits ['put_nowait', 'UnboundedQueue']
- KEEP `combine_file__1dshnds7` python-trio/trio::__aiter__ src/trio/_core/_unbounded_queue.py — F2P5+P2P_in0=5 (full P2P725), hits ['UnboundedQueue']
- KEEP `combine_file__5c1koagf` python-trio/trio::_get_batch_protected src/trio/_core/_unbounded_queue.py — F2P4+P2P_in1=5 (full P2P726), hits ['unpark', 'UnboundedQueue']
- KEEP `func_pm_class_rm_funcs__n83sv680` spulec/freezegun::freeze_time freezegun/api.py — F2P1+P2P_in4=5 (full P2P133), hits ['tick']
- KEEP `func_basic__pfmu1cyw` python-trio/trio::shield src/trio/_core/_run.py — F2P1+P2P_in5=6 (full P2P729), hits ['deadline']
- KEEP `func_pm_ctrl_shuffle__0g9hubd8` python-trio/trio::relative_deadline src/trio/_core/_run.py — F2P2+P2P_in4=6 (full P2P728), hits ['deadline']
- KEEP `pr_3096` python-trio/trio::sleep_until src/trio/_timeouts.py — F2P1+P2P_in5=6 (full P2P729), hits ['deadline']
- KEEP `lm_rewrite__w71ozq2u` python-trio/trio::sleep_forever src/trio/_timeouts.py — F2P3+P2P_in3=6 (full P2P727), hits ['deadline']
- KEEP `func_pm_ctrl_shuffle__5qwxjuzj` python-trio/trio::deadline src/trio/_core/_run.py — F2P1+P2P_in5=6 (full P2P729), hits ['deadline']
- KEEP `func_pm_ctrl_shuffle__7p96y4fc` python-trio/trio::effective_deadline src/trio/_core/_run.py — F2P1+P2P_in6=7 (full P2P729), hits ['deadline']
- KEEP `func_basic__udtech68` django/channels::? channels/auth.py — F2P4+P2P_in3=7 (full P2P73), hits ['flush']
- KEEP `func_basic__vcmscytv` python-trio/trio::deliver_next src/trio/_signals.py — F2P6+P2P_in2=8 (full P2P724), hits ['unpark']
- KEEP `lm_rewrite__gjk1btuw` django/daphne::handle_exception daphne/http_protocol.py — F2P1+P2P_in8=9 (full P2P62), hits ['chunk']
- KEEP `func_basic__lidcxt2h` spulec/freezegun::tick freezegun/api.py — F2P5+P2P_in6=11 (full P2P129), hits ['tick']
- KEEP `func_basic__4z2x9r32` spulec/freezegun::TickingDateTimeFactory.__call__ freezegun/api.py — F2P2+P2P_in9=11 (full P2P132), hits ['tick']
- KEEP `func_pm_ctrl_shuffle__e86ei5fl` spulec/freezegun::move_to freezegun/api.py — F2P1+P2P_in10=11 (full P2P133), hits ['tick', 'move_to']
- KEEP `pr_533` spulec/freezegun::FrozenDateTimeFactory.move_to freezegun/api.py — F2P2+P2P_in9=11 (full P2P132), hits ['tick', 'move_to']
- KEEP `lm_rewrite__5mw4n9fv` python-trio/trio::ParkingLotStatistics.remove_parking_lot_breaker src/trio/_core/_parking_lot.py — F2P1+P2P_in10=11 (full P2P729), hits ['parking_lot']
- KEEP `lm_rewrite__ojn25xiy` python-trio/trio::unpark src/trio/_core/_parking_lot.py — F2P1+P2P_in10=11 (full P2P729), hits ['parking_lot', 'unpark']
- KEEP `lm_rewrite__0kru2rjr` python-trio/trio::repark_all src/trio/_core/_parking_lot.py — F2P1+P2P_in10=11 (full P2P729), hits ['parking_lot', 'unpark', 'repark']
- KEEP `func_basic__hfap00ae` spulec/freezegun::StepTickTimeFactory.__call__ freezegun/api.py — F2P5+P2P_in8=13 (full P2P129), hits ['tick']
- KEEP `func_pm_ctrl_shuffle__w0b37fdo` spulec/freezegun::update_step_width freezegun/api.py — F2P5+P2P_in8=13 (full P2P129), hits ['tick']
- KEEP `func_pm_class_rm_funcs__0g5btsxu` python-trio/trio::__enter__ src/trio/_channel.py — F2P3+P2P_in10=13 (full P2P727), hits ['checkpoint']
- KEEP `lm_rewrite__el96ki99` python-trio/trio::start_guest_run src/trio/_core/_run.py — F2P2+P2P_in13=15 (full P2P728), hits ['tick', 'nursery', 'checkpoint']
- KEEP `lm_rewrite__hhhjl6pj` django/channels::flush channels/layers.py — F2P2+P2P_in15=17 (full P2P75), hits ['expired', 'put_nowait', 'flush', 'get_nowait']
- KEEP `func_basic__48ju6s77` python-trio/trio::_current_cookie_tick src/trio/_dtls.py — F2P6+P2P_in22=28 (full P2P724), hits ['tick']
- KEEP `func_basic__1uoillks` python-trio/trio::valid_cookie src/trio/_dtls.py — F2P16+P2P_in12=28 (full P2P714), hits ['tick']
- KEEP `func_pm_ctrl_shuffle__oc8ztnq0` python-trio/trio::remove_parking_lot_breaker src/trio/_core/_parking_lot.py — F2P3+P2P_in34=37 (full P2P727), hits ['parking_lot']
- KEEP `combine_file__e8p2iisa` python-trio/trio::_send_message_to_trio src/trio/_threads.py — F2P2+P2P_in45=47 (full P2P728), hits ['put_nowait', 'nursery']
- KEEP `lm_rewrite__nfjz5uuu` tornadoweb/tornado::_update_next tornado/ioloop.py — F2P1+P2P_in0=1 (full P2P1156), hits ['ioloop']
- KEEP `lm_rewrite__ck60lqcc` tornadoweb/tornado::_process_queue tornado/simple_httpclient.py — F2P2+P2P_in0=2 (full P2P1155), hits ['ioloop']

### Ordering notes (pilots + representatives)
- tornado `task_done` (rl4yuxyq) tornado/queues.py: putter admit ordering (full-vs-not-full with waiting putters); single-file total 3 (F2P1+P2P_in2). Same symbol as f5-replace KEEP, retained.
- daphne `handle_exception` (gjk1btuw) daphne/http_protocol.py:239 WebRequest.handle_reply http.response.start->body->finish lifecycle; single-file total 9 (F2P1+P2P_in8), full 63/7 files. PILOT (smallest full suite, best eligible chance).
- channels `flush` (hhhjl6pj) channels/layers.py:295 InMemoryChannelLayer._clean_expired expiry->discard->flush; single-file total 17 (F2P2+P2P_in15), full 77/10 files. PILOT (second smallest full).
- trio `_get_batch_protected/get_batch_nowait/put_nowait/empty/__aiter__/__anext__` (6 symbols, each total 5): UnboundedQueue FIFO produce A,B,C->consume batch [A,B,C]; e.g. eduyy9fy-family. Full 726-730/47 files structurally blocked on macOS (44 env P2P), single-file total 5 avoids env but normalize missing_files would reject subset — retained for selection diversity, not piloted for eligibility.
- trio MockClock `rate/jump/autojump_threshold/deadline_to_sleep_time` (each total 5): clock skew/scheduling ordering; same structural full-file block as above.
- trio timeouts `sleep_until/sleep_forever/shield/deadline/relative_deadline/effective_deadline` (total 6-7): deadline scheduling ordering; same block.
- freezegun `move_to/tick/__call__/freeze_time/update_step_width` (total 5-13): freeze->tick/move->read ordering; skip-prefix precheck required (f5-replace 1wzhexnu near-miss: required test_ticking_time denied by skipped test_ticking_time_clock). All 7 freezegun in 40 are single-F2P-file; precheck before dual.
- trio parking_lot `unpark/repark_all/remove_breaker` (total 11-37): FIFO pop/breaker ordering; full large, single small.
- trio dtls/threads/channel/guest (total 15-47): channel/threads/mock ordering; larger single totals but <50, retained as neutral/strong fill.

## REJECTS (excluded before 40, not counted)
- 17 BAD (read-only dedup from f5-replace triple-review-log.md): 6 dups (Frozen tick hq96g33n, Ticking tick 2l53f5xe, MockClock resync auc2kn82, relative_deadline 7ml3iktm/q8tz7b2d/gte8bpt6) +10 single-point (l3r7e4bf statistics -1, gvbsi1f0/i7x3of2o/i50xxyek nursery None/wrong-attr, u1sosk2w relative_deadline negated, d6l5ckls swapped, ls0ee38t inverted, d7uuw2nv thin wrapper, 62ckl2tv deregister, doej1tg5 init flags) +1 multi (j5yd2544 break_lot+statistics).
- 23 weak canon::sym (value-only swaps in websocket/auth/repr/hash/copy/decorate, e.g. OriginValidator.__init__ swap, validate_origin, login/logout) excluded from final 40 (0 weak retained).
- Full F2P+P2P<50: 0 candidates for 5 repos (proven by scan: min 63/70/128/684/1157). Single-file-overall (all F2P+P2P in 1 file): 0. Per-file small is the only viable small-P2P definition.
- Multi-file patches (74), F2P multi-file (190 multi-file but per-file small), no-timing-hit (417) excluded at scan.

## Family/snapshot status (40-list)
- Snapshots: same 5 shorts as f5-replace, all outside 75 banned (full-SHA confirmed for trio/freezegun; short-level disjoint for channels/daphne/tornado, banned has zero tornado snapshots). No extra ban (f5-replace pilots 0 eligible).
- Families: 40/40 distinct by canon::sym (38 strong-ordering +2 neutral tornado ioloop _update_next/_process_queue file-only hits but ioloop scheduling ordering retained as neutral). 0 dups, 0 single-point in final 40.
- Pilots for dual replay: daphne gjk1btuw + channels hhhjl6pj (smallest full suites 63/77, single-file totals 9/17, no known trio-env block; skip-prefix precheck first). Trio full-file structurally blocked (44 fails) so not piloted for eligibility; freezegun/tornado deferred after precheck.

## Audit addendum (2026-10-25, post-select patch re-read)
- Automated canon::sym dedup misclassified some trio queue patches (nearby def names). Manual re-read finds NEW single-point/multi among the 40 candidates (in addition to the 17 BAD already excluded):
  - REJECT `ly27fik4` (empty inverted `not _data` -> `len>0`) — single-point boolean.
  - REJECT `5ly62yyy` (qsize `len` -> `len-1`) — single-point value.
  - REJECT `3c4joxmg` (qsize+empty+_get_batch multi-hunk) — multi-function.
  - REJECT `5c1koagf` (__init__ flag + put_nowait insert) — multi-function (2 hunks, 2 symbols).
  - REJECT `4z2x9r32` (__init__ time_to_freeze/start swap) — single-point swap (value-only, no produce→consume→end).
  - REJECT `5mw4n9fv` (remove_breaker thin deregister) — no temporal sequence (same reason as f5-replace 62ckl2tv).
  - REJECT `1uoillks`/`48ju6s77` (dtls cookie tick) — no produce→consume→end ordering (cookie validation values).
  - REJECT `n83sv680` (freeze_time?) / `hfap00ae` needs full patch re-read; provisionally flagged NO-ORD pending second read (see scan table).
- Honest KEEP subset is therefore smaller than 40 (approx 25-30 true orderings, mirroring f5-replace ~15-18/40). No quota filling with single-point: gap reported. Pilots chosen from verified true orderings only: e86ei5fl (Ticking tick early-return, true) + 5c2wl96f (_get_batch_protected flag order, true; flag timing produce→consume→end).
- Precheck results: e86ei5fl BLOCKED (required P2P test_ticking_time shares prefix with skipped test_ticking_time_clock; confirmed skipped in upstream single-file run, 11 passed/1 skipped; no dual per precheck-first rule). 5c2wl96f PASSED single-file precheck (test_unbounded_queue.py 5 tests, 0 skipped) → dual running (run-002 wrong profile collection-error superseded by run-003 correct profile).
