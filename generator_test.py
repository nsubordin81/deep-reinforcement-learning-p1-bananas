def count(start=0, end=15):
    n = start
    while n < 10:
        if n / 3 == 1:
            return
        else:
            yield n
        n += 1


for n in count():
    print(n)

