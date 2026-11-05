#!/usr/bin/env python3
"""Compiles the generated Ray SPIR-V shaders with AMD RGA and writes a Markdown report.

Runs rga.exe (Radeon GPU Analyzer) in offline Vulkan SPIR-V mode over every *.spv
in internal/shaders/output and collects ISA size, register usage, spills, scratch,
LDS and a derived max-waves-per-SIMD occupancy estimate. RT pipeline stage shaders
(.rgen/.rchit/.rmiss/...) are skipped, they are not used by the VK backend.
"""

import argparse
import concurrent.futures
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

RT_STAGE_MARKERS = (".rgen.", ".rchit.", ".rmiss.", ".rint.", ".rahit.", ".rcall.")

DEFAULT_RGA = os.environ.get("RGA_EXE") or shutil.which("rga") or shutil.which("rga.exe")

# RDNA VGPR file per SIMD: 16K 32-bit regs, allocation granularity 8, wave32;
# hardware cap is 16 concurrent waves per SIMD.
MAX_WAVES_PER_SIMD = 16

print_mutex = threading.Lock()


def log(msg):
    with print_mutex:
        print(msg, flush=True)


def waves_per_simd(vgprs):
    if vgprs <= 0:
        return MAX_WAVES_PER_SIMD
    alloc = -(-vgprs // 8) * 8
    return min(MAX_WAVES_PER_SIMD, (16 * 1024) // (32 * alloc))


def is_rt_stage(name):
    return any(m in name for m in RT_STAGE_MARKERS)


def run_rga(rga_exe, spv_path, tmp_dir, asic, stem, timeout):
    out_csv = os.path.join(tmp_dir, "stats_" + stem)
    cmd = [rga_exe, "-s", "vk-spv-offline", "-c", asic,
           "--comp", spv_path, "-a", out_csv]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout after %d s" % timeout

    # RGA renames the analysis file to <asic>_<stem>_<stage>.csv next to the
    # requested path; glob for it to be safe against naming variations.
    found = None
    pattern = os.path.join(tmp_dir, "stats_" + stem)
    base = os.path.dirname(pattern)
    for f in os.listdir(base):
        if "stats_" + stem in f and f.endswith(".csv"):
            found = os.path.join(base, f)
            break
    if found is None:
        # Drop amdllpc crash backtrace lines ("0x... N byte(s)") from the error.
        err = " | ".join(l for l in (proc.stdout + "\n" + proc.stderr).splitlines()
                         if "byte(s)" not in l).strip()
        return None, "no stats CSV (exit %d): %s" % (
            proc.returncode, err[-300:] if err else "no output")

    with open(found, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, "empty stats CSV"
    return rows[0], None


def summarize_err(err, cap=180):
    # Keep only the meaningful parts of LLPC/amdllpc dumps: the failing pass
    # context line and the exception line.
    parts = [p.strip() for p in err.split("|") if p.strip()]
    keep = []
    for i, p in enumerate(parts):
        if p.lower().startswith(("exception", "llvm error", "error:")) or \
           p.startswith(("1.", "2.", "3.")) or "Running pass" in p:
            keep.append(p)
    text = " ".join(keep[-2:]) if keep else err
    if len(text) > cap:
        tail = text[-cap:]
        tail = tail[tail.find(" ") + 1:]  # don't start mid-word
        text = "..." + tail
    return text


def analyze_one(rga_exe, spv_path, tmp_dir, asic, idx, timeout):
    name = os.path.basename(spv_path)
    stem = "%03d_%s" % (idx, re.sub(r"\W+", "_", name[:-4]))
    row, err = run_rga(rga_exe, spv_path, tmp_dir, asic, stem, timeout)
    if err is None:
        num = lambda k: int(row.get(k) or 0)
        stats = {
            "name": name,
            "spv_size": os.path.getsize(spv_path),
            "isa_size": num("ISA_SIZE"),
            "vgprs": num("USED_VGPRs"),
            "sgprs": num("USED_SGPRs"),
            "vgpr_spills": num("VGPR_SPILLS"),
            "sgpr_spills": num("SGPR_SPILLS"),
            "scratch": num("SCRATCH_MEM"),
            "lds": num("USED_LDS_BYTES"),
        }
        log("  ok   %s" % name)
        return stats, None
    log("  FAIL %s: %s" % (name, summarize_err(err, 100)))
    return None, (name, err)


def fmt(n):
    return "{:,}".format(n)


def make_report(results, failures, skipped, asic, rga_version, rga_exe, src_desc):
    results.sort(key=lambda s: s["isa_size"], reverse=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# Ray shader stats (%s)" % asic)
    lines.append("")
    lines.append("Generated %s by `Ray/scripts/rga_shader_stats.py`, RGA %s, "
                 "source: `%s/*.spv` (offline SPIR-V mode, release builds only)."
                 % (ts, rga_version or "?", src_desc))
    lines.append("")
    lines.append("| # | Shader | SPV, B | ISA, B | VGPRs | SGPRs | VGPR spills "
                 "| SGPR spills | Scratch, B/thread | LDS, B | Max waves/SIMD | Occ. |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, s in enumerate(results, 1):
        waves = waves_per_simd(s["vgprs"])
        occ = 100 * waves // MAX_WAVES_PER_SIMD
        lines.append("| %d | `%s` | %s | %s | %d | %d | %d | %d | %d | %s | %d | %d%% |"
                     % (i, s["name"], fmt(s["spv_size"]), fmt(s["isa_size"]),
                        s["vgprs"], s["sgprs"], s["vgpr_spills"], s["sgpr_spills"],
                        s["scratch"], fmt(s["lds"]), waves, occ))
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append("- Max waves/SIMD is the VGPR-bound estimate: RDNA caps at %d "
                 "wave32/SIMD, VGPRs allocate in units of 8, VGPR file is 16K "
                 "regs/SIMD. Approximate, LDS and dispatch size ignored."
                 % MAX_WAVES_PER_SIMD)
    lines.append("- Offline-mode numbers come from AMD's standalone compiler "
                 "(same back end as the driver) and may differ slightly from "
                 "live-driver ISA.")
    lines.append("- `%d` RT pipeline stage shaders (.rgen/.rchit/.rmiss) were "
                 "skipped." % len(skipped))
    if failures:
        lines.append("- `%d` shaders failed to compile:" % len(failures))
        for name, err in failures:
            lines.append("  - `%s`: %s" % (name, summarize_err(err)))
    lines.append("")
    return "\n".join(lines)


ROW_RE = re.compile(r"^\| \d+ \| `(.+)` \| ([\d,]+) \| ([\d,]+) \| (\d+) \| (\d+) "
                    r"\| (\d+) \| (\d+) \| (\d+) \| ([\d,]+) \| ")


def parse_report(path):
    # Re-read stats of an existing report for --retry (occupancy cols are
    # derived and recomputed at write time).
    results = []
    for line in open(path, encoding="utf-8"):
        m = ROW_RE.match(line)
        if not m:
            continue
        g = [int(x.replace(",", "")) for x in m.groups()[1:]]
        results.append({"name": m.group(1), "spv_size": g[0], "isa_size": g[1],
                        "vgprs": g[2], "sgprs": g[3], "vgpr_spills": g[4],
                        "sgpr_spills": g[5], "scratch": g[6], "lds": g[7]})
    return results


def get_rga_version(rga_exe):
    try:
        out = subprocess.run([rga_exe, "--version"], capture_output=True,
                             text=True, timeout=30).stdout
        m = re.search(r"Version: (\S+)", out)
        return m.group(1) if m else out.strip().splitlines()[0]
    except Exception:
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shaders_dir = os.path.normpath(
        os.path.join(script_dir, "..", "internal", "shaders", "output"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rga", default=DEFAULT_RGA,
                    help="path to rga.exe (default: RGA_EXE env or PATH lookup)")
    ap.add_argument("--dir", default=shaders_dir, help="folder with *.spv files")
    ap.add_argument("--out", default=None,
                    help="output .md file (default: internal/shaders/"
                         "shader_stats_<asic>.md, committed on purpose)")
    ap.add_argument("--asic", default="gfx1201", help="RGA target ASIC (e.g. gfx1201)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel rga processes (default: min(16, CPU count))")
    ap.add_argument("--filter", default=None,
                    help="only process shaders whose name contains this substring")
    ap.add_argument("--retry", nargs="?", const="", default=None, metavar="REPORT",
                    help="reuse stats from an existing report (default: the "
                         "output file) and only (re)compile shaders missing from it")
    ap.add_argument("--timeout", type=int, default=300, help="per-shader timeout, s")
    args = ap.parse_args()
    if args.jobs is None:
        args.jobs = min(16, os.cpu_count() or 4)

    if not args.rga or not (os.path.isfile(args.rga) or shutil.which(args.rga)):
        print("rga.exe not found (add it to PATH, set RGA_EXE, or use --rga)")
        return 1
    if not os.path.isdir(args.dir):
        print("shader dir not found: %s" % args.dir)
        return 1

    spv_files = sorted((f for f in os.scandir(args.dir)
                        if f.name.endswith(".spv") and f.is_file()),
                       key=lambda f: f.name)
    skipped = [f.name for f in spv_files if is_rt_stage(f.name)]
    compute = [f.path for f in spv_files if not is_rt_stage(f.name)]
    if args.filter:
        compute = [p for p in compute if args.filter in os.path.basename(p)]

    default_out = os.path.normpath(os.path.join(
        script_dir, "..", "internal", "shaders",
        "shader_stats_%s.md" % args.asic))
    out_path = args.out or default_out

    results, failures = [], []
    if args.retry is not None:
        retry_src = args.retry or out_path
        old = parse_report(retry_src) if os.path.isfile(retry_src) else []
        if not old:
            print("--retry: no reusable stats in %s, doing a full run" % retry_src)
        else:
            wanted = {os.path.basename(p) for p in compute}
            results = [r for r in old if r["name"] in wanted]
            for r in results:  # spv sizes may have changed since
                p = os.path.join(args.dir, r["name"])
                if os.path.isfile(p):
                    r["spv_size"] = os.path.getsize(p)
            done = {r["name"] for r in results}
            compute = [p for p in compute if os.path.basename(p) not in done]
            print("Reusing %d shaders from %s, (re)compiling %d...\n"
                  % (len(done), retry_src, len(compute)))

    print("RGA: %s (%s)" % (args.rga, get_rga_version(args.rga)))
    print("Compiling %d compute shaders for %s (%d RT-stage skipped)...\n"
          % (len(compute), args.asic, len(skipped)))

    with tempfile.TemporaryDirectory(prefix="rga_stats_") as tmp:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(analyze_one, args.rga, p, tmp, args.asic, i,
                                args.timeout)
                    for i, p in enumerate(compute)]
            for fut in concurrent.futures.as_completed(futs):
                stats, err = fut.result()
                if err:
                    failures.append(err)
                else:
                    results.append(stats)

    try:
        src_desc = os.path.relpath(args.dir).replace(os.sep, "/")
    except ValueError:
        src_desc = args.dir
    report = make_report(results, failures, skipped, args.asic,
                         get_rga_version(args.rga), args.rga, src_desc)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n%d ok, %d failed, %d skipped -> %s"
          % (len(results), len(failures), len(skipped), out_path))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
