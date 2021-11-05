import pandas
from collections import Counter


a = ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
letter_counts = Counter(a)
df = pandas.DataFrame.from_dict(letter_counts, orient='index')
df.plot(kind='bar')
