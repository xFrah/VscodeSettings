def combinations(bases: list[int]):
    res = [[i] for i in range(bases[0])]
    periods = []
    for i in range(len(bases)):
        periods.append()


def get_comb_count(bases, index):
    m = 1
    for b in bases[:index]:
        m *= b
    return m


combinations([2, 5, 2])
