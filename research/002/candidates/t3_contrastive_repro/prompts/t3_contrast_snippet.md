### Contrastive repro (T3): make the repro answer the issue
1. Before claiming repro success, answer: does it contain the trigger? Which single input flips normal->abnormal? What exact difference (type/value/format/branch) is expected? Where seen (branch/output)?
2. If green unexpectedly, check the trigger axis first, e.g.: null value vs string-"null" annotation; bytes vs UploadFile parameter; single-line markup without header separator that never enters table path. Add at most one minimal contrast pair, not more same-class scripts.
3. Same input + same output + no source change twice -> next step must add information or make a local edit; no third identical rerun.
4. Dependency/fixture/import error is not repro success; keep reasoning from source, do not fix env for validation.
5. Temp repros stay in sandbox temp dir, never in final patch. No LLM judge, no reference answers.
