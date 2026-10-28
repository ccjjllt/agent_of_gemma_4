# SFT task-prefix sharing: fixed-cohort CPU mechanism

Independent read of current r4 encoded SHA5555b374… reproduces805rows,100tasks,2,054,401 input tokens and78,148 selected next-token targets. Each task's longest row totals344,887tokens. Exactly705rows are strict prefixes of their task's longest row. All100 mismatches are action0: generation inserts the prompt-only empty thought control token IDs[100,45518,107,101], decoded as `<|channel>thought\n<channel|>`, while full history omits that block. Removing it in memory aligns the first action, but the actual first-action training row must keep it. It has zero label weight yet changes the conditioning prefix.

Among the705 matching rows, selected absolute target positions never overlap. A prospective representation could keep each first action and union the remaining masks onto its task's longest row:200rows and476,745 input tokens (131,858 first-action tokens plus344,887 longest-row tokens), with all78,148 targets retained. This representation was built only in memory; recommended data, manifests and trainers remain unchanged. Counts are not a31B runtime or memory speedup measurement.

Actual real tiny Gemma4ForConditionalGeneration CPU experiment used one real8-action cohort, Mistune PR337. It kept the actual lengths[1212,1453,1629,2037,2383,2534,2695,2742], all-one attention masks and absolute0..L-1 positions. Input CT IDs were mapped one-to-one to602 small-vocabulary IDs (vocab603 including reserved pad0); no token position was dropped. The FP32/dropout0 model has2 causal hybrid sliding/full layers, window1024 and37,568 parameters; no optimizer or parameter update. This tests causal/gradient mechanics, not31B/NF4 token semantics.

|Same fixed task cohort|Original8 rows|First action + merged longest|
|---|---:|---:|
|Rows|8|2|
|Input tokens|16,685|3,954|
|Selected targets|541|541 (8+533)|
|Token-pooled loss|6.407942958176136|6.407943107187748|

Unreduced selected-token CE values were identical. Pooled loss difference was1.490116119e-7; maximum absolute gradient difference across27 tensors was1.676380634e-7, relative gradient L2 difference9.602384549e-7. Parameter hash stayed unchanged. These are the numerical reduction/backward differences of the same causal objective, not bitwise gradient equality.

The algebra requires the exact same per-target causal prefix, attention/positions, no stochastic dropout, disjoint target positions and one shared selected-token denominator for the same fixed cohort. Source action0 remains separate. Current single/FSDP trainers already pool selected tokens within each actual accumulation window. Their random row minibatches/windows would change if805rows became200; so would Adam/scheduler timing and sequence sampling weights. This experiment does not establish end-to-end training equivalence or authorize a dataset replacement.

Actual data/source/config fingerprints, per-task census, tiny configuration and numeric results are in the adjacent JSON. No production source, data export, recommendation or training configuration changed. No31B/NF4/GPU,optimizer,formal training,oracle,download,heldout or postsubmit rerun occurred.
