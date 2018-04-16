class dictlist(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return dictlist({key: value[index] for key, value in dict.items(self)})
    
    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value
    
    def append(self, d):
        for key, value in d.items():
            if not(key in dict.keys(self)):
                dict.__setitem__(self, key, [])
            dict.__getitem__(self, key).append(value)