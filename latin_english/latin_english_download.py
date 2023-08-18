from datasets import load_dataset
from datasets import load_from_disk


dataset = load_dataset("grosenthal/latin_english_translation",split="train")

# save
dataset.save_to_disk('./data/latin_english_train')

# test
dataset_test = load_dataset("grosenthal/latin_english_translation",split="test")
dataset_test.save_to_disk('./data/latin_english_test')

