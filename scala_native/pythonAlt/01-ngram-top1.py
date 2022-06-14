# import fileinput

max_w = ""
max_y = 0
max_c = 0

# for line in fileinput.input():
for line in open("d1.txt").readlines():
    w, y, c, _ = line.split("\t")
    c = int(c)
    if c > max_c:
        max_w = w
        max_y = int(y)
        max_c = c

print(f"{max_w}, {max_y}, {max_c}")
