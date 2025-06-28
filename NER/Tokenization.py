from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Any, Tuple
from exception import MyException


class TokenizationProcessor:
    def __init__(self, raw_data: List[Dict], tokenizer: PreTrainedTokenizerBase):
        """
        Args:
            raw_data (list): List of dicts with 'text' and 'entities'.
            tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer.
        """
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.label_list = []
        self.label2id = {}
        self.id2label = {}
        self.tokenized_dataset = None

    def build_label_mappings(self):
        """Build BIO tag list and label-id mappings from entity labels."""
        try:
            unique_labels = set()
            for sample in self.raw_data:
                for _, _, label in sample["entities"]:
                    unique_labels.add(label)

            self.label_list = ["O"]
            for label in sorted(unique_labels):
                self.label_list.extend([f"B-{label}", f"I-{label}"])

            self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}

        except Exception as e:
            raise MyException(f"Error while building label mappings: {e}")

    def prepare_dataset(self) -> Dataset:
        """Convert raw data into Hugging Face Dataset with start-end label format."""
        try:
            fixed = []
            for sample in self.raw_data:
                fixed_entities = [
                    {"start": start, "end": end, "label": label}
                    for start, end, label in sample["entities"]
                ]
                fixed.append({
                    "text": sample["text"],
                    "entities": fixed_entities
                })
            return Dataset.from_list(fixed)
        except Exception as e:
            raise MyException(f"Error while preparing dataset: {e}")

    def tokenize_and_align(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize input text and align labels to tokens (BIO tagging)."""
        try:
            text = example["text"]
            entities = example["entities"]

            # Character-level label list
            char_labels = ["O"] * len(text)
            for entity in entities:
                start, end, label = entity["start"], entity["end"], entity["label"]
                if 0 <= start < end <= len(text):
                    char_labels[start] = f"B-{label}"
                    for i in range(start + 1, end):
                        char_labels[i] = f"I-{label}"

            # Tokenization with offsets
            tokenized = self.tokenizer(text, truncation=True, return_offsets_mapping=True)
            labels = []
            for start_char, end_char in tokenized["offset_mapping"]:
                if start_char == end_char:
                    labels.append(-100)
                else:
                    span_labels = char_labels[start_char:end_char]
                    if all(l == "O" for l in span_labels):
                        labels.append(self.label2id["O"])
                    else:
                        labels.append(self.label2id.get(span_labels[0], self.label2id["O"]))

            tokenized["labels"] = labels
            tokenized.pop("offset_mapping")
            return tokenized

        except Exception as e:
            raise MyException(f"Error during tokenization and alignment: {e}")

    def process(self) -> Tuple[int, Dict[str, int], Dict[int, str], Dataset]:
        """
        Full pipeline: label mapping, dataset conversion, tokenization.

        Returns:
            num_labels (int): Total number of unique BIO labels.
            label2id (dict): BIO label to ID mapping.
            id2label (dict): ID to BIO label mapping.
            tokenized_dataset (Dataset): Tokenized and labeled dataset.
        """
        self.build_label_mappings()
        dataset = self.prepare_dataset()
        self.tokenized_dataset = dataset.map(self.tokenize_and_align)
        return len(self.label_list), self.label2id, self.id2label, self.tokenized_dataset
