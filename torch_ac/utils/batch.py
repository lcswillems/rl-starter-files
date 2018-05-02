import random

def batchify(l, batch_size):
    if batch_size == 0:
        batch_size = len(l)

    indexes = list(range(len(l)))
    random.shuffle(indexes)
    
    return [l[i:i+batch_size] for i in range(0, len(l), batch_size)]