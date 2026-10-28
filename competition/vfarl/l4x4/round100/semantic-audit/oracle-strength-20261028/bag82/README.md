# Fixed100 index82: Bag multiplicity oracle gap

2026-10-28. The oracle weakness probes in this directory are research evidence and must never be teacher/canonical/SFT/RL inputs. Wrong source is reconstructable from the pinned original snapshot plus [WRONG-NOT-TRAINING patch](WRONG-ORACLE-PROBE-NOT-TRAINING.patch); [reconstruction metadata](reconstruction.json) binds both. The separately executed [silent-preload fresh correct teacher r2](multiplicity-teacher-r2/README.md) contains only the correct Counter repair and its own actual tool/oracle observations. R1 is retained intact; its dependency-preload timing output entered real tool observations, so r2 independently reruns the teacher with silent imports rather than deleting observed text.

The current/original task refers to character frequencies. Correct Bag distance uses Counter multiplicity and the maximum unmatched count. Converting each Counter to set preserves the displayed qwe/anagram/disjoint cases but returns0 for aab/ab and aaab/ab, where the same source-bound reviewed repair returns1 and2. The existing teacher observes abb/abc=1; this happens to equal the set result too.

Original full oracle:29 target files,5 F2P+422 P2P=427 required nodes,430 actual expanded JUnit cases. Under the explicit dependency-preload runtime:

| Production variant | Original430 cases | Official resolved |
|---|---|---|
| Original mutant | 5 fail /425 pass | false |
| Correct reviewed repair | 430 pass | true |
| Wrong set deduplication | 430 pass | true |

[Full results, stdout and JUnit](preload-r1/oracle-observations.json) retain the actual unmodified verify_task outcomes. The task/oracle/required nodes/Hypothesis deadlines/seeds were not changed. Only a task-only [.pth](preload-r1/00_vfarl_dependency_preload.pth) imports NumPy and installed external-comparator modules before pytest timing; it calls no distance/target function. Its actual startup import costs were9.001/8.383/8.112seconds for the three processes and are reported separately. All16original declared package versions were recovered from existing compatible macOS CP311 cache, without downloads or hostADK changes. This explicit runtime is not presented as unchanged original profile13 or Kaggle/Linux equivalence.

The [first non-preloaded attempt](oracle-observations.json) remains intact: baseline8fail/reference3fail, where the three remaining failures were unrelated first-call Hypothesis200ms deadline FlakyFailures; wrong was not run after that failed control. No old reward is rewritten.

This is a direct multiset main-contract counterexample accepted by the complete original oracle in the stated runtime. Root can consider pausing future original-oracle RL for82 while preserving the correct SFT and historical records; this report changes no eligibility or course itself. The fresh correct teacher independently strengthens the inline source-bound observation, without consuming the wrong patch.
