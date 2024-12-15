import conllu


def load_data(path):
    # load the conllu dataset
    with open(path, encoding='utf-8') as f:
        data = conllu.parse(f.read())

    # extract headlines (since a headline can have more than one sentence)
    headlines = []
    for i, sentence in enumerate(data):
        if sentence.metadata["sent_id"] == "0":
            headlines.append(data[i:i+1])
        else:
            headlines[-1].append(sentence)
    
    return headlines