## Allocation type vs bandwidth
| alloc_type | access | bandwidth_gb_s | percent_peak |
|---|---|---|---|
| cudaMalloc | device | 204.46 | 74.89 |
| cudaMallocManaged | first | 162.40 | 59.49 |
| cudaMallocManaged | second | 163.15 | 59.76 |
| cudaHostAlloc | device | 176.33 | 64.59 |
| malloc+cudaHostRegister | device | 103.88 | 38.05 |
