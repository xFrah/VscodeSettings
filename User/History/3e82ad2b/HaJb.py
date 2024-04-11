def combinations(bases: list[int]):
    index = 0
    total = get_comb_count(bases, index)
    res = [[] for _ in range(total)]
    for b in bases:
        for i in range(b):
            sublist = [i] * (total // b)
            print(sublist)
            for j in res[::b]:
            j.append

    print(res)


def get_comb_count(bases, index):
    m = 1
    for b in bases[index:]:
        m *= b
    return m


combinations([2, 5])
