def combinations(bases: list[int]):
    periods = [get_comb_count(bases, i) for i in range(len(bases))]
    bases_periods = zip(bases, periods)
    res = [[(i % b) * b for b, period in bases_periods] for i in range(get_comb_count(bases, len(bases)))]
    print(periods)


def get_right_number(i, b):
    return (i % b) * b


def get_comb_count(bases, index):
    m = 1
    for b in bases[:index]:
        m *= b
    return m


combinations([2, 5, 2])
