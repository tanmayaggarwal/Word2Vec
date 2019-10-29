# read in the text file

def load_data(file_path):
    with open(file_path) as f:
        text = f.read()

    # print out the first 100 characters
    print(text[:100])

    return text 
