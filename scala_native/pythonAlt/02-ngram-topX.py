import fileinput
import time

max_w = ""
max_y = 0
max_c = 0

data = []

rs = time.perf_counter_ns()
for line in fileinput.input():
    w, y, c, _ = line.split("\t")
    data.append((w, y, int(c)))
rl = time.perf_counter_ns() - rs
print(f"reading took {rl/1000**2 :.4f} ms")

ss = time.perf_counter_ns()
data.sort(key=lambda e: -e[2])
sl = time.perf_counter_ns() - ss

print(f"sorting took {sl/1000**2 :.4f} ms")
for i in range(10):
    w, y, c = data[i]
    print(f"{w}, {y}, {c}")
