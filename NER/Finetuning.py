# Import finetuning library
from transformers import AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from NER.Data_processing import NERDataPreparer
from NER.Tokenization import TokenizationProcessor

# Prepare the data 
prep = NERDataPreparer("Give the filepath")
prep_data = prep.prepare()


# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenization Preprocessing
tokp = TokenizationProcessor(prep_data,tokenizer)
num_labels,label2id,id2label,tokenized_dataset = tokp.process()



# Model loading
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list),
    label2id=label2id,
    id2label=id2label
)

# Applying LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],
    task_type="TOKEN_CLS"
)

# Setting Training Argument
training_args = TrainingArguments(
    output_dir="./lora-ner-project",
    per_device_train_batch_size=32,
    num_train_epochs=20, 
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="no",
    eval_strategy="no",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    fp16=True
)

# Settingup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)
# Now train it!
trainer.train()