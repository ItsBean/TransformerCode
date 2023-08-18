from datasets import load_dataset
from datasets import load_from_disk
from transformers import BertTokenizer

# load train and test datasets
dataset_train = load_from_disk('./data/latin_english_train')
dataset_test = load_from_disk('./data/latin_english_test')




# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Tokenize the dataset
def tokenize_data(example):
    latin_encoded = tokenizer(example['la'], return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    english_encoded = tokenizer(example['en'], return_tensors='pt', max_length=128, truncation=True,
                                padding='max_length')

    return {
        'latin_input_ids': latin_encoded['input_ids'][0],
        'latin_attention_mask': latin_encoded['attention_mask'][0],
        'english_input_ids': english_encoded['input_ids'][0],
        'english_attention_mask': english_encoded['attention_mask'][0]
    }


tokenized_train_dataset = dataset_train.map(tokenize_data)
tokenized_test_dataset = dataset_test.map(tokenize_data)

# save
tokenized_train_dataset.save_to_disk('./data/latin_english_train_tokenized')
tokenized_test_dataset.save_to_disk('./data/latin_english_test_tokenized')


# print('# tokenized_train_dataset sentence id', tokenized_train_dataset)
# ['id', 'la', 'en', 'file', 'latin_input_ids', 'latin_attention_mask', 'english_input_ids', 'english_attention_mask']

print('id:', tokenized_train_dataset[0])
print('# len of la', len(tokenized_train_dataset[0]['la']))
print('# len of en', len(tokenized_train_dataset[0]['en']))
print('# len of latin_input_ids', len(tokenized_train_dataset[0]['latin_input_ids']))
print('# len of english_input_ids', len(tokenized_train_dataset[0]['english_input_ids']))
# len of la 114
# len of en 113
# len of latin_input_ids 128
# len of english_input_ids 128

print(tokenized_train_dataset[0]['latin_input_ids'][113:])
