def combinations(bases: list[int]):
    res = [[i] for i in range(bases[0])]
    for b in bases[1:]:
        for i in range(b):
            for e in res:
                res.append(e + [i])
    print(res)


def get_comb_count(bases, index):
    m = 1
    for b in bases[index:]:
        m *= b
    return m


combinations([2, 5, 2])
