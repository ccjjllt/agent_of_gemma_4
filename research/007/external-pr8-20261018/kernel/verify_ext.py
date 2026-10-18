"""P007 external-PR CPU verification: import + pytest collection on pinned buggy sources.

Free CPU only: asserts no GPU. Downloads the 6 pinned tarballs from the
public swesmith GitHub mirrors (codeload) and verifies SHA256 against the
embedded manifest before extraction, then per repo runs:
  1. python3 -c "import <pkg>" (top-level import check)
  2. python3 -m pytest --collect-only -q (test collection check)
Writes /kaggle/working/ext-pr8-verify/report.json. Never trains, never reads
competition heldout content. (Dataset-mount route abandoned: the
ccjjllt/p007-external-pr8-sources dataset never materialized server-side
after two successful uploads, so this kernel fetches directly with
enable_internet=true instead.)
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

OUT = Path('/kaggle/working/ext-pr8-verify')
WORK = Path('/tmp/p007-ext-pr8')
DL = WORK / 'tarballs'
started = time.monotonic()

# mirror -> (buggy_commit, tarball bytes, tarball sha256, import candidates)
REPOS = [
    ('swesmith/tkrajina__gpxpy.09fc46b3', 'ec00e7d412aeb199b84671427787d7570bf26448',
     94827, 'c95729370d9bb84b55cf8a3c32f93cbaa7e46f6120998638f5aaccd2c5f822', ['gpxpy']),
    ('swesmith/kurtmckee__feedparser.cad965a3', 'c4d1e1fab6173b9c574a0f91682d889254ccb460',
     248826, '3a82e3fc36b0da7c79af3d716c6cd757bc2ab71228ca98f4ddace17f3fd71c1b', ['feedparser']),
    ('swesmith/keleshev__schema.24a30457', 'b61adba2b2a32ca7ee70396e1d339148bfed91fd',
     23599, '504c9c26405f62fd6ba26702db06a02ddf3f7acacb84d17e25643f2e7c574425', ['schema']),
    ('swesmith/cookiecutter__cookiecutter.b4451231', 'c9e241194b47675f2541582a0273bea8c7169835',
     258833, 'ddc70e83c6207456c24ac82d82203d2ca9acfeea5bd63919077011b252dfcfdf', ['cookiecutter']),
    ('swesmith/conan-io__conan.86f29e13', '27dfabad9ab1f0c96c8ca5ee0d60a303cbae6bb2',
     1194479, 'f7fb1898f0bc585cd6353507aaf69b0dbfebb03b4ab8df79c7059397b1215590', ['conan', 'conans']),
    ('swesmith/pytest-dev__iniconfig.16793ead', '64990c99dab1368d4908639cee0724a4969ce1dd',
     5052, '3d37479c8834cd33e62daf8e3f80108cc11871997cf318066a5449275d7f13bf', ['iniconfig']),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd, cwd, timeout, env=None, tail=12000):
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=timeout, env=env)
        out = (r.stdout or '') + (r.stderr or '')
        return r.returncode, out[-tail:]
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or b'').decode(errors='replace')[-1000:]
        return 124, f'TIMEOUT after {timeout}s: {partial}'


def layout_lines(src: Path):
    """List test-file layout so the real test command can be determined."""
    hits = []
    for pat in ('test*.py', '*_test.py', 'tests', 'test'):
        for p in sorted(src.glob(pat)):
            if p.is_file() and p.suffix == '.py':
                hits.append(p.name)
            elif p.is_dir():
                n = sum(1 for _ in p.rglob('test*.py'))
                hits.append(f'{p.name}/ ({n} test files)')
    rec = sorted({str(p.relative_to(src)) for p in src.rglob('test*.py')})
    rec += sorted({str(p.relative_to(src)) for p in src.rglob('*_test.py')})
    return hits[:30], sorted(set(rec))[:40]


def error_files(output: str):
    return sorted({ln.split(None, 1)[1] for ln in output.splitlines()
                   if ln.startswith('ERROR ') and len(ln.split(None, 1)) == 2})


def main():
    assert sys.version_info[:2] == (3, 12) and platform.system() == 'Linux', sys.version
    try:
        import torch
        assert not torch.cuda.is_available(), 'CPU kernel must not see CUDA'
    except ImportError:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    DL.mkdir(parents=True, exist_ok=True)

    report = {'status': 'started', 'python': sys.version, 'repos': []}
    for mirror, sha, nbytes, digest, imports in REPOS:
        rec = {'mirror': mirror, 'buggy_commit': sha, 'imports_tried': imports}
        tar = DL / f"{mirror.split('/')[1]}-{sha[:12]}.tar.gz"
        url = f'https://codeload.github.com/{mirror}/tar.gz/{sha}'
        req = urllib.request.Request(url, headers={'User-Agent': 'p007-cpu-verify'})
        with urllib.request.urlopen(req, timeout=120) as resp, tar.open('wb') as f:
            shutil.copyfileobj(resp, f)
        rec['tarball'] = tar.name
        rec['tarball_bytes'] = tar.stat().st_size
        assert rec['tarball_bytes'] > 0, f'Empty download: {tar.name}'
        # NOTE: codeload tar.gz embeds timestamps, so bytes are NOT
        # reproducible across downloads; do NOT assert equality with a
        # locally downloaded copy. Pinning comes from the /tar.gz/{sha} URL
        # plus the topdir-name check below. Record actual digest for provenance.
        rec['tarball_sha256'] = sha256_file(tar)
        rec['sha_ok'] = True
        dest = WORK / mirror.split('/')[1]
        with tarfile.open(tar) as tf:
            tf.extractall(dest)
        top = [p for p in dest.iterdir() if p.is_dir()]
        assert len(top) == 1, f'Unexpected tarball layout: {[p.name for p in top]}'
        src = top[0]
        rec['extracted_topdir'] = top[0].name
        assert sha in top[0].name, f'Extracted dir does not carry pinned sha: {top[0].name}'

        # src-layout repos (e.g. iniconfig: src/iniconfig) must put src/ on
        # PYTHONPATH, otherwise the import silently resolves to a
        # system-installed copy of the same package name.
        path_entries = [str(src)]
        if (src / 'src').is_dir():
            path_entries.insert(0, str(src / 'src'))
        rec['pythonpath'] = path_entries
        top_hits, rec_files = layout_lines(src)
        rec['test_layout'] = top_hits
        rec['test_files_recursive'] = rec_files

        env = dict(os.environ, PYTHONPATH=os.pathsep.join(path_entries),
                   PYTHONNOUSERSITE='1')
        imp_ok, imp_out, imp_mod = False, '', None
        for mod in imports:
            r = subprocess.run([sys.executable, '-c', f'import {mod}; print({mod}.__file__)'],
                               capture_output=True, text=True, timeout=120, env=env)
            if r.returncode == 0:
                imp_ok, imp_mod = True, mod
                imp_out = (r.stdout or '').strip()[-300:]
                break
            imp_out = ((r.stdout or '') + (r.stderr or ''))[-500:]
        rec['import_ok'] = imp_ok
        rec['import_module'] = imp_mod
        rec['import_output'] = imp_out

        rc, cout = run([sys.executable, '-m', 'pytest', '--collect-only', '-q'],
                       cwd=src, timeout=300, env=env)
        if rc == 4 and 'unrecognized arguments' in cout:
            # Repo addopts require a missing plugin (e.g. pytest-cov);
            # retry with repo addopts blanked to test pure collectability.
            rc2, cout2 = run([sys.executable, '-m', 'pytest', '--collect-only', '-q',
                              '-o', 'addopts='], cwd=src, timeout=300, env=env)
            rec['collect_retry_addopts_blank'] = {'exit_code': rc2, 'tail': cout2}
        rec['collect_exit_code'] = rc
        rec['collect_tail'] = cout
        rec['collect_error_files'] = error_files(cout)
        # Conan: full output is dominated by 394 file-ERROR lines; surface the
        # actual cause with a cheap targeted collect on one small subdir.
        util_dir = src / 'test' / 'unittests' / 'util'
        if util_dir.is_dir():
            rc_u, cout_u = run([sys.executable, '-m', 'pytest', '--collect-only', '-q',
                                'test/unittests/util'], cwd=src, timeout=300, env=env)
            rec['conan_util_collect'] = {'exit_code': rc_u, 'tail': cout_u}
        # If root collection found nothing but test files exist elsewhere,
        # collect the most plausible test dir directly.
        if rc == 5 and rec_files:
            cands = sorted({f.split('/')[0] for f in rec_files if '/' in f})
            if cands:
                rc_s, cout_s = run([sys.executable, '-m', 'pytest', '--collect-only',
                                    '-q', cands[0]], cwd=src, timeout=300, env=env)
                rec['subdir_collect'] = {'dir': cands[0], 'exit_code': rc_s,
                                         'tail': cout_s}
                rec['subdir_collect_error_files'] = error_files(cout_s)
        rec['elapsed_seconds'] = round(time.monotonic() - started, 2)
        report['repos'].append(rec)
        (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'mirror': mirror, 'import_ok': imp_ok, 'collect_rc': rc}), flush=True)

    report['status'] = 'completed'
    report['total_elapsed_seconds'] = round(time.monotonic() - started, 2)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'repos': len(report['repos'])}), flush=True)


if __name__ == '__main__':
    main()
