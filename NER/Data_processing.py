import json
from exception import MyException

class NERDataPreparer:
    def __init__(self, input_path, output_path):
        """
        Initialize the NERDataPreparer with input and output paths.

        Args:
            input_path (str): Path to the Label Studio JSON export file.
            output_path (str): Path to save the processed NER data.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.ner_data = []
        self.skipped_entries = 0

    def load_data(self):
        """Loads raw data from the input JSON file."""
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise MyException(f"Failed to load input file: {e}")

    def extract_entities(self, raw_data):
        """Processes the raw data to extract entities."""
        for idx, example in enumerate(raw_data):
            try:
                text = example["data"]["text"]
                annotations = example["annotations"][0]["result"]
                entities = []

                for ann in annotations:
                    value = ann["value"]
                    label = value["labels"][0]
                    start = value["start"]
                    end = value["end"]

                    if 0 <= start < end <= len(text):
                        entities.append([start, end, label])
                    else:
                        raise MyException(f"Invalid span: {start}-{end} in text of length {len(text)}")

                if entities:
                    self.ner_data.append({
                        "text": text,
                        "entities": entities
                    })

            except Exception:
                self.skipped_entries += 1
                continue

    def save_data(self):
        """Saves the extracted entity data to the output path."""
        try:
            with open(self.output_path, "w", encoding="utf-8") as out:
                json.dump(self.ner_data, out, indent=2)
        except Exception as e:
            raise MyException(f"Failed to write output file: {e}")

    def prepare(self):
        """Full pipeline: load, extract, save."""
        raw_data = self.load_data()
        self.extract_entities(raw_data)
        self.save_data()

        return {
            "status": "success",
            "extracted_samples": len(self.ner_data),
            "skipped_entries": self.skipped_entries,
            "output_file": self.output_path,
            "data": self.ner_data
        }
