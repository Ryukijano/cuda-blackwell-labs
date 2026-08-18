# Project 21: Key Findings

- The `total=` counters in `/proc/pressure/memory` are cumulative microseconds of stall time.
- `some` is more useful on a lightly loaded system; `full` indicates severe contention.
- Correlating PSI with GPU read bandwidth shows when the system is bandwidth-bound vs. stalled on other memory-system resources.
