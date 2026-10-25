from pathlib import Path
import os, sys, json, subprocess, ctypes, ctypes.util
def record_capabilities(out):
    probes = {'uid': os.getuid(), 'euid': os.geteuid(),
              'process_status': Path('/proc/self/status').read_text(),
              'libseccomp': ctypes.util.find_library('seccomp')}
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    probes['landlock'] = {'abi': abi, 'errno': ctypes.get_errno() if abi < 0 else 0}
    commands = [
        ['unshare', '--mount', '/bin/true'],
        ['unshare', '--pid', '--fork', '/bin/true'],
        ['unshare', '--net', '/bin/true'],
        [sys.executable, '-c', 'import tempfile,os; d=tempfile.mkdtemp(prefix="vfarl-chroot-"); '
         'os.chdir(d); os.chroot(d); print("chroot succeeded",os.getcwd())'],
        [sys.executable, '-c', 'import ctypes; l=ctypes.CDLL(None,use_errno=True); '
         'r=l.prctl(38,1,0,0,0); print({"no_new_privs":r,"errno":ctypes.get_errno()})'],
    ]
    probes['commands'] = []
    for command in commands:
        try:
            result = subprocess.run(command, text=True, capture_output=True)
            probes['commands'].append({'command': command, 'returncode': result.returncode,
                                       'stdout': result.stdout, 'stderr': result.stderr})
        except Exception as exc:
            probes['commands'].append({'command': command, 'error': str(exc)})
    (out / 'isolation-capabilities.json').write_text(json.dumps(probes, indent=2) + '\n')
    print(json.dumps({'phase': 'isolation_capabilities', **probes}), flush=True)
out=Path("/kaggle/working/capabilities"); out.mkdir(); record_capabilities(out)
