"""Read-only macOS memory sampling alongside Higgs capacity telemetry."""
import ctypes
import re
import subprocess


class RusageInfoV2(ctypes.Structure):
    # macOS SDK sys/resource.h, rusage_info_v2 (160 bytes).
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
        'wired_size', 'resident_size', 'phys_footprint', 'proc_start_abstime',
        'proc_exit_abstime', 'child_user_time', 'child_system_time',
        'child_pkg_idle_wkups', 'child_interrupt_wkups', 'child_pageins',
        'child_elapsed_abstime', 'diskio_bytesread', 'diskio_byteswritten')]


def memory_sample(pid):
    lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    info = RusageInfoV2()
    assert ctypes.sizeof(info) == 160
    if lib.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(2), ctypes.byref(info)):
        raise OSError(ctypes.get_errno(), 'proc_pid_rusage')
    vm = subprocess.check_output(['vm_stat'], text=True, timeout=3)
    page_size = int(re.search(r'page size of (\d+) bytes', vm)[1])
    pages = {key.strip(): int(value) for key, value in re.findall(r'^([^:\n]+):\s+(\d+)\.', vm, re.M)}
    return {'pid': pid, 'physical_footprint_bytes': info.phys_footprint,
            'resident_bytes': info.resident_size, 'wired_bytes': info.wired_size,
            'process_start_abstime': info.proc_start_abstime,
            'system_page_size': page_size, 'system_vm_counters': pages,
            'swap_usage': subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True, timeout=3).strip()}
