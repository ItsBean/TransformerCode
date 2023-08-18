import spacy
spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
