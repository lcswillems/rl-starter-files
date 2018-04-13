def concat_to_dict(d_to_concat, d):
    for key in d.keys():
        if not(key in d_to_concat.keys()):
            d_to_concat[key] = []
        d_to_concat[key] += d[key]

def append_to_dict(d_to_append, d):
    for key in d.keys():
        if not(key in d_to_append.keys()):
            d_to_append[key] = []
        d_to_append[key].append(d[key])