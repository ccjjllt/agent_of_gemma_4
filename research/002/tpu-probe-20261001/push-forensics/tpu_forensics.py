"""TPU forensics probe (Stage 1, no model load, no pip install).
Goals: record actual machine type, chip count, per-chip memory, preinstalled
version fingerprints, and weight config quantization fields (JSON only).
Budget: must finish in minutes. All output goes to stdout + evidence JSON.
"""
import datetime
import glob
import json
import os
import platform
import subprocess
import sys


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": p.returncode,
                "stdout": p.stdout[-4000:], "stderr": p.stderr[-2000:]}
    except Exception as e:
        return {"cmd": " ".join(cmd), "rc": -1, "stdout": "", "stderr": repr(e)}


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    ev = {"probe": "tpu-forensics-20261001", "started_utc": t0.isoformat(),
          "requested_machine_shape": "TpuV6E8"}
    print(f"[forensics] start {ev['started_utc']}", flush=True)

    ev["python"] = sys.version
    ev["platform"] = platform.platform()
    ev["env_tpu"] = {k: v for k, v in os.environ.items()
                     if any(s in k for s in ("TPU", "TPU_", "XLA", "JAX", "ACCEL", "KAGGLE"))}

    # CPU / mem baseline
    ev["lscpu"] = run(["lscpu"])
    ev["meminfo_head"] = run(["sh", "-c", "head -n 20 /proc/meminfo"])
    ev["dev_accel"] = run(["sh", "-c", "ls -la /dev/ | grep -i -E 'accel|tpu|vfio' || echo NO_ACCEL_DEV_MATCH"])
    ev["ls_kaggle_input"] = run(["sh", "-c", "ls -la /kaggle/input/ 2>&1 | head -n 50"])

    # Preinstalled package fingerprints (no install here)
    ev["pip_freeze_tpu"] = run(["sh", "-c",
        "pip list 2>/dev/null | grep -i -E 'jax|libtpu|tpu|vllm|torch|transformers|compressed|llm' || echo NO_MATCH"])
    for pkg in ("jax", "jaxlib", "libtpu", "vllm", "tpu-inference"):
        try:
            __import__("importlib.metadata").metadata.version
            import importlib.metadata as md
            try:
                ev[f"ver_{pkg}"] = md.version(pkg)
            except Exception:
                ev[f"ver_{pkg}"] = "NOT_INSTALLED"
        except Exception as e:
            ev[f"ver_{pkg}"] = f"ERR {e!r}"

    # JAX device topology (real chip count / memory, no model load)
    try:
        import jax
        devs = jax.devices()
        ev["jax_backend"] = jax.default_backend()
        ev["jax_devices"] = [str(d) for d in devs]
        ev["jax_ndevices"] = jax.device_count()
        ev["jax_local_ndevices"] = jax.local_device_count()
        print(f"[forensics] jax backend={ev['jax_backend']} n={ev['jax_ndevices']} local={ev['jax_local_ndevices']}", flush=True)
        for d in devs:
            print(f"[forensics] device: {d}", flush=True)
        # per-device memory stats if available
        try:
            for d in devs[:4]:
                s = jax.local_devices()
                break
            ev["jax_mem_note"] = "see device lines above; full mem stats need model load"
        except Exception as e:
            ev["jax_mem_note"] = repr(e)
    except Exception as e:
        ev["jax_error"] = repr(e)
        print(f"[forensics] JAX ERROR: {e!r}", flush=True)

    # Weight config JSON only (no weight load): verify quant fields vs PR#3653 scope
    try:
        cands = glob.glob("/kaggle/input/**/config.json", recursive=True)
        ev["config_candidates"] = cands[:10]
        print(f"[forensics] config candidates: {cands[:10]}", flush=True)
        for c in cands[:3]:
            try:
                with open(c) as f:
                    cfg = json.load(f)
                q = cfg.get("quantization_config", {})
                ev.setdefault("quant_configs", {})[c] = {
                    "quant_method": q.get("quant_method"),
                    "format": q.get("format"),
                    "weight_sym": (q.get("weight_block_quant") or q.get("config_groups") or q) if isinstance(q, dict) else str(type(q)),
                    "raw_keys": sorted(q.keys()) if isinstance(q, dict) else [],
                    "raw": json.dumps(q)[:3000],
                }
                print(f"[forensics] quant_config @ {c}: {json.dumps(ev['quant_configs'][c], ensure_ascii=False)[:1500]}", flush=True)
                # also record arch/context hints
                ev.setdefault("model_meta", {})[c] = {
                    "architectures": cfg.get("architectures"),
                    "model_type": cfg.get("model_type"),
                    "max_position_embeddings": cfg.get("max_position_embeddings"),
                    "head_dim": cfg.get("head_dim"),
                }
            except Exception as e:
                ev.setdefault("config_errors", {})[c] = repr(e)
    except Exception as e:
        ev["weight_scan_error"] = repr(e)

    # tokenizer/chat-template fingerprint (cheap, no model load)
    try:
        cands = glob.glob("/kaggle/input/**/tokenizer_config.json", recursive=True)
        ev["tokenizer_candidates"] = cands[:5]
    except Exception as e:
        ev["tokenizer_scan_error"] = repr(e)

    t1 = datetime.datetime.now(datetime.timezone.utc)
    ev["finished_utc"] = t1.isoformat()
    ev["elapsed_s"] = (t1 - t0).total_seconds()
    out = "/kaggle/working/forensics_evidence.json"
    with open(out, "w") as f:
        json.dump(ev, f, indent=2, ensure_ascii=False)
    print(f"[forensics] done elapsed={ev['elapsed_s']:.1f}s wrote {out}", flush=True)


if __name__ == "__main__":
    main()
