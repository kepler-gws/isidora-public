



# https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
def listofdict_to_dictoflist(lod):
    return {k: [dic[k] for dic in lod] for k in lod[0]}

def continuous_binary_search(tolerance, min_val, max_val, test_func):
    required_val = None
    while (abs(max_val - min_val) > tolerance):
        required_val = (min_val + max_val) / 2
        if test_func(required_val):
            max_val = required_val
        else:
            min_val = required_val
    # Value that makes test_func eval to false.
    required_val = min_val
    assert not test_func(required_val)
    return required_val