def first_fit(items, bin_capacity):
    bins = []
    for item in items:
        for bin in bins:
            if bin_capacity - bin >= item:
                bin.append(item)
                break
        else:
            bins.append([item])
    return bins
