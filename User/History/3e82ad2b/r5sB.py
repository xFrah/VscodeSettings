def combinations(bases: list[int]):
    res = [[i] for i in range(bases[0])]
    periods = [get_comb_count(bases, i) for i in range(len(bases))]
    for 
    print(periods)


def get_comb_count(bases, index):
    m = 1
    for b in bases[:index]:
        m *= b
    return m


combinations([2, 5, 2])
