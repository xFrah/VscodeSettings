def combinations(bases: list[int]):
    index = 0
    total = get_comb_count(bases, index)
    res = [[] for _ in range(total)]
    for b in bases:
        for i in range(b):
            for j in res[i::b]:
                j.append(i)

    print(res)


def get_comb_count(bases, index):
    m = 1
    for b in bases[index:]:
        m *= b
    return m


combinations([2, 5])
