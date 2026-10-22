# Train-next source CPU experiment

This is an independent train-only CPU execution, following the completed P015 panel CPU run. It makes no model requests and does not use dev or heldout tasks.

The data owner freezes `research/015/data/train-next-material` first. Only then are the opaque `data/train-next-material.bin` archive hash and file-member hashes embedded in `prepare.py`, with the final source input recorded in `input-manifest.json`. The root owner chooses the private dataset identifier, fills `kernel-metadata.json` dataset_sources, and performs the upload and CPU launch. This directory does not start remote jobs.

The experiment preserves each task's original Git HEAD and runs the source-provided criterion on buggy and reference states. Existing custom F2P/P2P commands, pytest plugins and source-required pytest arguments are retained. Tasks with `environment.install_editable=true` install their checked-out source using offline pip with `--no-index --no-deps --no-build-isolation -e`, then verify module origin points into the actual task checkout. It does not create tests or assert new behavior.

Material extraction, source checkouts, builds and per-task virtual environments stay in `/tmp/p015-train-next-prevalidation`. Each task's temporary checkout/environment is deleted after execution. `/kaggle/working/p015-train-next-prevalidation` contains only logs, actual results, summary, provenance, and one byte-identical `prepared-train-next-material.bin`. The `.bin` extension preserves tar bytes while avoiding platform recursive extraction of inner snapshot archives.

The eight-task input is now frozen from the data owner's final material manifest `c0b1590da817f077a9c2dfe48ea67066b63b8c8f7e56dad7edb8cf00d7799171`. The uploaded opaque archive contains 245 verified files, 69,711,476 bytes, SHA256 `3d7f97a3b1e2d5843126fd9f4e961eadcdfb66bcbd32d25816314e391cfff240`. Dataset metadata uses `zhuating/p015-train-next-material` and the CPU kernel is `zhuating/p015-train-next-cpu`. Local evidence covers macOS Python3.12 only; no Linux result is recorded until root actually executes the carrier.
