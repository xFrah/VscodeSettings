def combinations(bases: list[int]):
    index = 0
    total = get_comb_count(bases, index)
    res = [[i] for i in range(bases[0])]
    for b in bases[1:]:
    print(res)


def get_comb_count(bases, index):
    m = 1
    for b in bases[index:]:
        m *= b
    return m


combinations([2, 5, 2])
