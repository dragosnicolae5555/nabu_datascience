from sortedcontainers import SortedDict
import itertools
d = SortedDict({(key1,key2):val for val,(key1,key2) in enumerate(itertools.product(range(4), range(4)))}) 
len(d)
upd = {(0,0):21}
print((0,0) in d._list)

d.update(upd)

print((0,0) in d._list)