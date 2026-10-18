# Windows compatibility fix report
- Date/time: 2026-10-18 11:35
- Tester:
- Original commit tested: f190881
- Branch: teammate/windows-validation
- Current HEAD (before commit): f190881 Initial commit: ROCm Performance Guard
- Operating system: Microsoft Windows 11 Home Single Language (build 10.0.26200)
- Python version: 3.13.0

## Results
- Before: 329 run, 324 passed, 5 failed
- After: 329 run, 326 passed, 3 failed

## Files changed
- .gitattributes (new): "experiments/results/** -text" so Git does not convert line endings in historical evidence files. This fixes test_parallel_experiment test_16 and test_transpose_experiment test_16, which hash the baseline file byte-for-byte.
- No tests, source code, expected hashes, or experiment results were edited.

## Remaining failures (not changed, awaiting owner's decision)
- test_hardware: test_host_memory_may_be_none_when_unavailable
- test_hardware: test_linux_visible_memory_separately_represented
- test_hardware: test_wsl_detection_in_current_environment
These tests assume Linux/WSL (/proc/meminfo and a WSL environment) and fail on native Windows. A platform-aware version of the tests was proposed but not applied.

## Evidence integrity
- Baseline experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json: 0 CRLF, SHA-256 ded6b85ff3ec29349018c25ae7e0d7aedcb3ea719380dc5dec57d1f31d2376ad (matches the expected hash).
- git diff for the file is empty (exit code 0) and its blob hash equals the index entry (fe77f7cd...). git status still lists it as modified, which appears to be a stale index cache after the file was recreated from Git's stored copy; the content is identical.
- The on-disk CRLF copy was replaced with the committed LF bytes by recreating it from Git's stored blob (no manual edits). Other files under experiments/results/ may still be CRLF on this machine; they are unchanged in Git.

## Not verified
- Linux/WSL behavior (suite not run in Ubuntu/WSL)
- AMD GPU performance and native ROCm/HIP compatibility

## Update: hardware tests made platform-aware
- Changed tests/test_hardware.py (adds import sys; updates 3 tests). On Linux the original strict assertions are kept (visible memory is a positive int, WSL v2 check inside WSL). On native Windows, Linux-only fields may be None, and WSL must be reported as False and match is_running_in_wsl().
- Final result on Windows 11, Python 3.13.0: 329 run, 329 passed, 0 failed (final_unit_tests.txt).
- Not verified: the Linux/WSL branches of these tests were not run on Ubuntu/WSL. Please run the suite there before merging.
- No source code, expected hashes, or experiment results were edited.
