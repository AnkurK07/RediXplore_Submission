from transformers import AutoModelForTokenClassification, AutoTokenizer

# model = AutoModelForTokenClassification.from_pretrained("kankur0007/BERT-NER-Projects", num_labels=9, ignore_mismatched_sizes=True)
# tokenizer = AutoTokenizer.from_pretrained("kankur0007/BERT-NER-Projects")

# model.save_pretrained("./saved_model")
# tokenizer.save_pretrained("./saved_model")

model = AutoModelForTokenClassification.from_pretrained("./saved_model",num_labels=9, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

