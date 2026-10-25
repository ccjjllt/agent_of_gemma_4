# F5-replace triple review log (2026-10-25)

Scope: Smith-py train `77cab905…13558e`, F5 event/stream/lifecycle timing only.
Banned: 75 snapshot_group_id (64 S600-common + 11 F1-new). Four competition repos excluded.
Each item: patch92856 single-function confinement + live-on-3.12 + temporal sequence (produce→consume→end) + single-point/escape rejection.

## KEEP (true F5 ordering, single-function, live) — 40-list

### aio-libs/async-timeout — DROPPED (dead code, see rejects)

### django/channels (snapshot a144b4b8, 2)
- KEEP `lm_rewrite__hhhjl6pj` channels/layers.py:295 `InMemoryChannelLayer._clean_expired` — message/group expiry → group discard → flush ordering; F2P2/P2P75 single-hunk.
- KEEP `func_pm_class_rm_funcs__s49jtv5s` channels/layers.py:359 `InMemoryChannelLayer.group_send` — group fan-out (join→send→receive); single-function deletion, F2P4/P2P73.

### django/daphne (snapshot 32ac73e1, 1)
- KEEP `lm_rewrite__gjk1btuw` daphne/http_protocol.py:239 `WebRequest.handle_reply` — http.response.start→body→finish lifecycle; single-function rewrite, F2P1/P2P62.

### spulec/freezegun (snapshot 5f171db0, 8)
- KEEP `func_basic__1wzhexnu` FrozenDateTimeFactory.tick — int seconds / float seconds-vs-minutes / timedelta forward-vs-backward; F2P9/P2P125.
- REJECT `func_pm_ctrl_shuffle__hq96g33n` — DUP family FrozenDateTimeFactory.tick (same symbol as 1wzhexnu).
- KEEP `func_basic__ny5bncew` FrozenDateTimeFactory.move_to — freeze→move→read ordering.
- KEEP `func_basic__lidcxt2h` TickingDateTimeFactory.__call__ — ticking read ordering.
- KEEP `func_pm_ctrl_shuffle__w0b37fdo` StepTickTimeFactory.tick — step tick ordering.
- KEEP `func_basic__cmdzx0wb` StepTickTimeFactory.__call__ — return-old-then-tick vs return-new ordering.
- KEEP `func_pm_ctrl_shuffle__e86ei5fl` TickingDateTimeFactory.tick — ticking factory ordering.
- REJECT `func_basic__2l53f5xe` — DUP family TickingDateTimeFactory.tick (same symbol as e86ei5fl).
- (pilot) freezegun-tick = 1wzhexnu fully prepared + replaying.

### python-trio/trio (snapshot cfbbe2c1, 28 in 40-list, mixed verdicts)
- KEEP `func_basic__eduyy9fy` UnboundedQueue.put_nowait — FIFO append vs LIFO insert(0); produce A,B,C→consume batch order; F2P4. (pilot, prepared + replaying)
- KEEP `lm_rewrite__d8w3osa0` Deadlines.next_deadline — stale-entry skip vs raw heap top; schedule→reschedule→query; F2P1.
- KEEP `func_basic__2ixvpkyc` Run.run_in_system_nursery.in_trio_thread — thread→trio outcome Value-vs-Error + context; F2P1.
- KEEP `lm_rewrite__ojn25xiy` ParkingLot._pop_several — FIFO pop with broken/zero/inf edge ordering; F2P1.
- KEEP `func_basic__158aks7r` ParkingLot.repark — repark order reversed + data/owner; F2P3.
- KEEP `func_pm_ctrl_shuffle__oc8ztnq0` add_parking_lot_breaker — check-then-register vs register-then-check; F2P3.
- KEEP `func_basic__h9e4lzy3` MockClock.current_time — +1.0 clock skew across produce→tick→read; F2P1.
- KEEP `func_basic__cj2elgta` CancelStatus._mark_abandoned — abandon propagation + children clear; F2P1.
- KEEP `func_basic__413eetye` MockClock._try_resync_autojump_threshold — guest tick scheduling condition; F2P4 (borderline, sequence via guest/non-guest).
- KEEP `lm_rewrite__ymdp5ij1` CancelScope.deadline setter — relative→absolute warning vs immediate cancel; F2P4 (borderline timing).
- KEEP `lm_rewrite__f1win0au` CancelScope.__exit__ — enter→cancel→exit exception filter ordering; F2P3 (large rewrite, single-function).
- KEEP `func_basic__7mwul5hr` Runner.spawn_system_task — system_task flag + context propagation; F2P2 (borderline flags, sequence via context var).
- KEEP `func_basic__sb29kzkn` NurseryManager.__enter__ — nursery enter ordering; F2P1.
- REJECT `lm_rewrite__auc2kn82` — DUP family MockClock._try_resync (same symbol as 413eetye).
- REJECT `func_basic__l3r7e4bf` ParkingLot.statistics — single-point count -1.
- REJECT `func_basic__gvbsi1f0` Task.parent_nursery — single-point identity None.
- REJECT `func_pm_class_rm_funcs__j5yd2544` ParkingLot.break_lot — multi-function hunk (break_lot + statistics deletion straddle).
- REJECT `func_basic__u1sosk2w` CancelScope.relative_deadline — single-point negation (also DUP 4x).
- REJECT `func_basic__7ml3iktm` — DUP relative_deadline + single-point.
- REJECT `func_pm_ctrl_invert_if__q8tz7b2d` — DUP relative_deadline + single-point.
- REJECT `lm_rewrite__gte8bpt6` — DUP relative_deadline + single-point.
- REJECT `func_basic__i7x3of2o` Task.eventual_parent_nursery — single-point None.
- REJECT `func_basic__i50xxyek` Nursery.parent_task — single-point wrong attribute.
- REJECT `func_basic__d6l5ckls` Runner.current_statistics — single-point swapped/negated values.
- REJECT `func_basic__ls0ee38t` CancelStatus.effective_deadline — single-point inverted deadline logic (borderline propagation, but value-only; reject for strict sequence).
- REJECT `func_basic__d7uuw2nv` ParkingLot.repark_all — single-point? (needs full patch; reject pending: likely thin wrapper, no ordering).
- REJECT `func_pm_ctrl_shuffle__62ckl2tv` remove_parking_lot_breaker — thin deregister, no temporal sequence; reject.
- REJECT `func_basic__doej1tg5` CancelScope.__attrs_post_init__ — init flags, no produce→consume→end; reject.
- REJECT `func_basic__l3r7e4bf` (already), `func_basic__gvbsi1f0` (already).
- KEEP `func_basic__eduyy9fy`, `d8w3osa0`, `2ixvpkyc`, `ojn25xiy`, `158aks7r`, `oc8ztnq0`, `h9e4lzy3`, `cj2elgta` (8 trio KEEP in pilot scope; rest pending re-pick).

### tornadoweb/tornado (snapshot d5ac65c1 short, full pending rate-limit, 1 in 40-list)
- KEEP `lm_rewrite__rl4yuxyq` Queue.get_nowait — putter admit ordering (full-vs-not-full with waiting putters); F2P1/P2P1156 large-suite risk noted.

## REJECTS (structural, not counted)
- DEAD-CODE (Python <3.11 else-branch on 3.12): all 3 aio-libs/async-timeout in v1 40-list (`etiyc452` timeout_at new_event_loop/deadline-5, `xhowq34y` shift deadline-delay, `6u2agdmd` reject guard) — upstream 5.0.1 uses asyncio.Timeout on 3.12; else-branch never executes (matches old-F5 dropped 4 async-timeout). Replaced.
- SINGLE-POINT values/booleans: async `38edt9p9` deadline -1.0, `h3hd9tj5` expired inverted, trio `ci26jlfk` effective_deadline negated, `l3r7e4bf` waiting -1, `gvbsi1f0`/`i7x3of2o`/`i50xxyek` nursery None/wrong-attr, `u1sosk2w`/`7ml3iktm`/`q8tz7b2d`/`gte8bpt6` relative_deadline negated, `d6l5ckls` statistics swapped, starlette `s9zxwiy7` flag=False, `f5g7uil7` content reversed, `ly27fik4` empty inverted.
- ESCAPE/SSE impersonation: 0 in 40-list (pre-filteredHTMLMLServerSentEvent/text-event-stream/sse_/html.escape).
- BAD-PATH ../dev/tests: mido 361 rows excluded at scan (prefix unmatchable under file-scope JUnit, matches old-F5 dropped mido-5).
- MULTI-FUNCTION hunks: freezegun pr_533/pr_539 (tick+move_to deletions), trio j5yd2544 (break_lot+statistics), channels pr? (none in 40-list).
- DUP families (same canon::symbol): relative_deadline 4x, Frozen tick 2x, Ticking tick 2x, MockClock resync 2x — keep one each, rest replaced/pending.
- LARGE-SUITE risk (P2P>600, skip-prefix collisions like F2-replace): trio 26 (P2P 700+), tornado 1 (P2P1156), starlette excluded; pilot covers 2 large-suite (trio put-order P2P726? actually 4/726? eduyy9fy F2P4/P2P726? and d8w3osa0 F2P1/P2P729) to test skip behavior; gap notes risk.

## Family/snapshot status (40-list v2)
- Snapshots: 6 distinct shorts (async dropped): trio cfbbe2c1, freezegun 5f171db0, channels a144b4b8, daphne 32ac73e1, tornado d5ac65c1, (+ async removed). Full SHAs resolved for 5 (tornado pending rate-limit, short-level disjoint confirmed: banned has zero tornado snapshots).
- Families: 34/40 distinct by canon::symbol (6 dups listed above). After triple-review KEEP strict ordering, ~15-18 true F5 remain; 6 dup + ~10 single-point/multi must be replaced with new ordering picks (tornado ioloop/process/iostream/websocket/http1connection, daphne server/ws, trio channel/threads/mock ordering, freezegun move_to/__call__ ordering). Gap reported; no quota filling with single-point.
