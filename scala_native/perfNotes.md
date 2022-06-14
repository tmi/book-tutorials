Benchmarks for top1 ngram:
- native1: 111s
- bash (sort -k 3 -r -n | head -n 1) 112s
- python naive: 45s
- mypyc
- cython

Benchmarks for topX ngram, 10m rows
- native1: 7m15s
- bash (sort -k 3 -r -n | head -n 10) 9.6s
- python naive: 9.0s

# post note -- bad perf of native1 is caused by the heap alloc due to vararg list constructed at every invoc
