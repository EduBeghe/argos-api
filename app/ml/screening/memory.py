from pathlib import Path


def parse_linux_peak_resident_bytes(status: str) -> int | None:
    """Parse Linux VmHWM, reported in KiB, into bytes."""

    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024
                except ValueError:
                    return None
    return None


def approximate_peak_resident_bytes(
    traced_peak: int, *, status_path: Path = Path("/proc/self/status")
) -> int:
    """Prefer OS peak RSS so native engine allocations are represented."""

    try:
        status = status_path.read_text(encoding="ascii")
    except OSError:
        return traced_peak
    return parse_linux_peak_resident_bytes(status) or traced_peak
