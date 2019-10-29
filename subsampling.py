# subsampling the data to remove words that show up often (e.g., the, of, for) but provide little to no context to the nearby words
# subsampling helps remove the noise in the data and get faster training and better representation
threshold = 1e-5
word_counts = Counter(int_words)
# print(list(word_counts.items())[0])

total
