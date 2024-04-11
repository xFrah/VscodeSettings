def combinations(bases: list[int]):
    periods = [get_comb_count(bases, i) for i in range(len(bases))]
    res = []
    print(periods)


def get_comb_count(bases, index):
    m = 1
    for b in bases[:index]:
        m *= b
    return m


combinations([2, 5, 2])
