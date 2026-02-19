import os
import re
import subprocess
import sys


def list_devices():
    if sys.platform == "win32":
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -Property Name"],
            capture_output = True, text = True, timeout = 15,
        )
        names = [
            line.strip() for line in result.stdout.strip().splitlines()
            if line.strip() and line.strip() != "Name" and not line.strip().startswith("----")
        ]
    elif sys.platform == "darwin":
        names = _list_devices_macos()
    else:
        names = _list_devices_linux()
    print("GPU devices:")
    for i, name in enumerate(names):
        vram = detect_device_memory(True, i)
        if vram is not None:
            print(f"  {i}: {name} ({vram / 1e9:.1f} GB)")
        else:
            print(f"  {i}: {name} (unknown VRAM)")


def _list_devices_linux() -> list[str]:
    names: list[str] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            names.extend(line.strip() for line in result.stdout.strip().splitlines() if line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Warning: nvidia-smi failed: {e}", file=sys.stderr)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                match = re.search(r"Card series:\s*(.+)", line)
                if match:
                    names.append(match.group(1).strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Warning: rocm-smi failed: {e}", file=sys.stderr)
    return names


def _list_devices_macos() -> list[str]:
    names: list[str] = []
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                match = re.search(r"Chipset Model:\s*(.+)", line)
                if match:
                    names.append(match.group(1).strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Warning: system_profiler failed: {e}", file=sys.stderr)
    return names


def _detect_gpu_memory_macos() -> list[int | None]:
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        n_gpus = len(re.findall(r"Chipset Model:", result.stdout)) if result.returncode == 0 else 1
        return [mem] * n_gpus
    except (OSError, subprocess.TimeoutExpired, AttributeError) as e:
        print(f"Warning: failed to detect macOS GPU memory: {e}", file=sys.stderr)
        return []


gguf_quant_preference = [
    "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q6_k", "q3_k_l",
    "q3_k_m", "q8_0", "q4_0", "q5_0", "q3_k_s", "q2_k",
    "fp16", "f16",
]


def _detect_gpu_memory_windows() -> list[int | None]:
    script = (
        "$pnps = (Get-CimInstance Win32_VideoController).PNPDeviceID;"
        "$base = 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}';"
        "$map = @{};"
        "foreach ($i in 0..20) {"
        "  $p = \"$base\\\" + $i.ToString('D4');"
        "  if (Test-Path $p) {"
        "    $props = Get-ItemProperty $p -ErrorAction SilentlyContinue;"
        "    if ($props.MatchingDeviceId -and $props.'HardwareInformation.qwMemorySize') {"
        "      $map[$props.MatchingDeviceId] = $props.'HardwareInformation.qwMemorySize'"
        "    }"
        "  }"
        "};"
        "foreach ($pnp in $pnps) {"
        "  $found = $false;"
        "  foreach ($key in $map.Keys) {"
        "    if ($pnp -like \"$key*\") { Write-Host $map[$key]; $found = $true; break }"
        "  };"
        "  if (-not $found) { Write-Host 'None' }"
        "}"
    )
    result = subprocess.run(
        ["powershell", "-Command", script],
        capture_output=True, text=True, timeout=15,
    )
    values: list[int | None] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line and line != "None":
            try:
                values.append(int(line))
            except ValueError:
                values.append(None)
        else:
            values.append(None)
    return values


_gpu_memory_cache: list[int | None] | None = None


def _detect_gpu_memory_linux() -> list[int | None]:
    values: list[int | None] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line:
                    try:
                        values.append(int(line) * 1024 * 1024)
                    except ValueError:
                        values.append(None)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Warning: nvidia-smi memory query failed: {e}", file=sys.stderr)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                match = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", line)
                if match:
                    values.append(int(match.group(1)))
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Warning: rocm-smi memory query failed: {e}", file=sys.stderr)
    return values


def detect_device_memory(gpu: bool, device: int) -> int | None:
    global _gpu_memory_cache
    if not gpu:
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["powershell", "-Command",
                     "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                    capture_output=True, text=True, timeout=10,
                )
                return int(result.stdout.strip())
            else:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, subprocess.TimeoutExpired, AttributeError) as e:
            print(f"Warning: failed to detect system memory: {e}", file=sys.stderr)
            return None
    try:
        if _gpu_memory_cache is None:
            if sys.platform == "win32":
                _gpu_memory_cache = _detect_gpu_memory_windows()
            elif sys.platform == "darwin":
                _gpu_memory_cache = _detect_gpu_memory_macos()
            else:
                _gpu_memory_cache = _detect_gpu_memory_linux()
        if device < len(_gpu_memory_cache):
            return _gpu_memory_cache[device]
    except (ValueError, OSError, subprocess.TimeoutExpired, AttributeError) as e:
        print(f"Warning: failed to detect GPU memory: {e}", file=sys.stderr)
    return None


def select_best_gguf(repo_id: str, memory_budget: int | None, kv_cache_reserve: int = 0) -> tuple[list[str], int]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    gguf_files = [
        (s.rfilename, s.size or 0)
        for s in info.siblings
        if s.rfilename.endswith(".gguf")
    ]

    if not gguf_files:
        raise ValueError(f"No .gguf files found in {repo_id}")

    groups: dict[str, list[tuple[str, int]]] = {}
    for filename, size in gguf_files:
        base = re.sub(r'-\d{5}-of-\d{5}\.gguf$', '.gguf', filename)
        groups.setdefault(base, []).append((filename, size))

    candidates = [
        (base, sum(s for _, s in files), len(files) == 1, files)
        for base, files in groups.items()
    ]

    if memory_budget is not None:
        budget = int(memory_budget * 0.6) - kv_cache_reserve
        fitting = [c for c in candidates if c[1] <= budget]
        if fitting:
            fitting.sort(key=lambda c: (-c[1], not c[2]))
            chosen = fitting[0]
        else:
            candidates.sort(key=lambda c: (c[1], not c[2]))
            chosen = candidates[0]
    else:
        def quant_rank(base: str) -> int:
            lower = base.lower()
            for i, quant in enumerate(gguf_quant_preference):
                if quant in lower:
                    return i
            return len(gguf_quant_preference)

        single = [c for c in candidates if c[2]]
        pool = single if single else candidates
        chosen = min(pool, key=lambda c: quant_rank(c[0]))

    base, total_size, is_single, files = chosen
    sorted_files = sorted(f[0] for f in files)
    return sorted_files, total_size


def _resolve_gpu_layers(gpu: bool, gpu_layers: int | None, device: int, model_size: int | None = None) -> int:
    if not gpu:
        return 0
    if gpu_layers is not None:
        return gpu_layers
    if model_size:
        vram = detect_device_memory(True, device)
        if vram:
            ratio = vram * 0.6 / model_size
            return -1 if ratio >= 1.0 else int(100 * ratio)
    return -1
