n = 3
tmp = ""
for i in range(1, n * 2):
    tmp = ""
    for j in range(i if i <= n else n - i % n ):
        tmp += f'{j + 1} '
    print(tmp)


