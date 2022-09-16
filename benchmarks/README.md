# Benchmarks for algorithms in PhDSE.jl

## Specs

```
julia> versioninfo()
Julia Version 1.8.1
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 8 × Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-13.0.1 (ORCJIT, cascadelake)
  Threads: 1 on 8 virtual cores
```

## Kalman filter

#### Predict step

```
BenchmarkTools.Trial: 14 samples with 1 evaluation.
 Range (min … max):  334.670 ms … 633.132 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     339.149 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   373.352 ms ±  90.258 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █
  █▅▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▁
  335 ms           Histogram: frequency by time          633 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
```

#### Correct step

```
BenchmarkTools.Trial: 1 sample with 1 evaluation.
 Single result which took 24.232 s (0.00% GC) to evaluate,
 with a memory estimate of 30.75 KiB, over 3 allocations.
```

## Square-root Kalman filter

#### Predict Step

```
BenchmarkTools.Trial: 5 samples with 1 evaluation.
 Range (min … max):  993.755 ms …   1.076 s  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):        1.012 s              ┊ GC (median):    0.00%
 Time  (mean ± σ):      1.029 s ± 37.519 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █   █        █                                   █         █
  █▁▁▁█▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁█ ▁
  994 ms          Histogram: frequency by time          1.08 s <

 Memory estimate: 70.31 MiB, allocs estimate: 6.
```

#### Correct step

```
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  2.965 s …  2.972 s  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.969 s             ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.969 s ± 5.211 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █                                                      █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  2.97 s        Histogram: frequency by time        2.97 s <

 Memory estimate: 277.98 MiB, allocs estimate: 8.
```

## Ensemble Kalman filter

[![enkf-nbviewer](https://img.shields.io/badge/EnKF_complexity-nbviewer-blue)](https://nbviewer.org/github/schmidtjonathan/PhDSE.jl/blob/main/benchmarks/EnKF_benchmark.ipynb)

#### Predict Step

```
BenchmarkTools.Trial: 80 samples with 1 evaluation.
 Range (min … max):  61.959 ms …  69.811 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     62.419 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   62.657 ms ± 941.783 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

      █▆ ▁▆       ▃   ▁
  ▄▇▇▆██▄██▇▇▄▇▄▁▆█▆▇▆█▆▄▄▁▁▁▄▇▁▁▁▁▁▄▄▄▁▁▄▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▁
  62 ms           Histogram: frequency by time         64.6 ms <

 Memory estimate: 16 bytes, allocs estimate: 1.
```

#### Correct step

```
BenchmarkTools.Trial: 18 samples with 1 evaluation.
 Range (min … max):  288.351 ms … 294.980 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     289.862 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   290.355 ms ±   1.866 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁▁ ▁ ▁█ ▁ ▁  ▁▁   ▁█       ▁ █                         ▁    ▁
  ██▁█▁██▁█▁█▁▁██▁▁▁██▁▁▁▁▁▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁█ ▁
  288 ms           Histogram: frequency by time          295 ms <

 Memory estimate: 16 bytes, allocs estimate: 1.
```

