import matplotlib.pyplot as plt

from nltk.corpus import brown
from collections import Counter


brown_tokens = brown.words()
brown_unique_tokens = set(brown_tokens)

brown_token_occurrences = Counter(brown_tokens)

# plt.plot(brown_token_occurrences)
