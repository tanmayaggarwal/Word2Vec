# helps pre-process the text to make the training easier

# converts punctuations to tokens (e.g., period is changed to <PERIOD>)
# removes all words that show up five or fewer times in the dataset to reduce issues due to noise in the data
# returns a list of words in the text

import utils

def pre_process(text):
    words = utils.preprocess(text)

    #print(words[:30])

    return words
