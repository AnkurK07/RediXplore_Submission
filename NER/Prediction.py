
import fitz  
import re
from typing import List, Dict, Any
from exception import MyException


# This can also taken out from Tokenization modulle but i won't include it in pipeline so i defined this manually.
label2id = {'O': 0, 'B-LOCATION': 1, 'I-LOCATION': 2, 'B-ORGANIZATION': 3, 'I-ORGANIZATION': 4, 'B-PROJECT': 5, 'I-PROJECT': 6, 'B-PROSPECT': 7, 'I-PROSPECT': 8}
id2label = {0: 'O', 1: 'B-LOCATION', 2: 'I-LOCATION', 3: 'B-ORGANIZATION', 4: 'I-ORGANIZATION', 5: 'B-PROJECT', 6: 'I-PROJECT', 7: 'B-PROSPECT', 8: 'I-PROSPECT'}


class ModelPrediction:
    def __init__(self, pdf_path: str, model, tokenizer, id2label=id2label):
        """
        Args:
            pdf_path (str): Path to the input PDF file.
            model: HuggingFace NER model.
            tokenizer: Corresponding tokenizer.
            id2label (dict): Mapping from label ids to string labels (e.g., B-PROJECT).
        """
        self.pdf_path = pdf_path
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label

    # Clean the text before feeding into model
    def clean_text(self, text: str) -> str:
        """Cleans text using regex rules for NER preprocessing."""
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = text.replace(' .', '.')
        text = re.sub(r"[^a-zA-Z0-9\s.,\'-]", '', text)
        return text.strip()
    
    # Extraction of text per page with some metadata
    def extract_text_per_page(self) -> List[Dict[str, Any]]:
        """Extracts cleaned text per PDF page."""
        try:
            doc = fitz.open(self.pdf_path)
            result = []
            for page_num, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")
                if raw_text.strip():
                    cleaned_text = self.clean_text(raw_text)
                    result.append({
                        "pdf_file": self.pdf_path.split("/")[-1],
                        "page_number": page_num,
                        "text": cleaned_text
                    })
            return result
        except Exception as e:
            raise MyException(f"Error reading PDF: {e}")

        
    # Do the prediction by the model
    def prediction(self, text: str) -> List[Dict[str, Any]]:
        """Runs token classification model on input text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

            entities = []
            current_entity = None

            for idx, pred_id in enumerate(predictions):
                label = self.id2label.get(pred_id, "O")
                if label == "O":
                    if current_entity:
                        current_entity["end"] = inputs.token_to_chars(idx - 1).end
                        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                        entities.append(current_entity)
                        current_entity = None
                elif label.startswith("B-"):
                    if current_entity:
                        current_entity["end"] = inputs.token_to_chars(idx - 1).end
                        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                        entities.append(current_entity)
                    current_entity = {
                        "label": label[2:],
                        "start": inputs.token_to_chars(idx).start,
                        "end": inputs.token_to_chars(idx).end
                    }
                elif label.startswith("I-") and current_entity:
                    current_entity["end"] = inputs.token_to_chars(idx).end

            if current_entity:
                current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                entities.append(current_entity)

            return entities
        except Exception as e:
            raise MyException(f"NER prediction failed: {e}")
        
    # Extracted Predicted Result in required json formet
    def extract_structured_jsonl(self, entities, text, pdf_file="filename.pdf", page_number=1):
        """Extracts structured info from PROJECT entities."""
        try:
            structured_data = []
            cleaned_text = self.clean_text(text)
            entities = sorted([e for e in entities if e["label"] == "PROJECT"], key=lambda x: x["start"])

            # Merge adjacent entities
            merged_entities = []
            i = 0
            while i < len(entities):
                merged_entity = entities[i].copy()
                j = i + 1
                while j < len(entities) and entities[j]["label"] == "PROJECT" and \
                        entities[j]["start"] - merged_entity["end"] < 5:
                    merged_entity["end"] = entities[j]["end"]
                    merged_entity["text"] = text[merged_entity["start"]:merged_entity["end"]]
                    j += 1
                merged_entities.append(merged_entity)
                i = j

            for ent in merged_entities:
                project_name = ent["text"].strip()
                if len(project_name) < 5 or project_name.lower() in {
                    "the", "and", "for", "with", "from", "project", "report"
                }:
                    continue

                cleaned_entity = self.clean_text(project_name)
                start_cleaned = cleaned_text.find(cleaned_entity, max(0, ent["start"] - 20))
                if start_cleaned == -1:
                    continue
                end_cleaned = start_cleaned + len(cleaned_entity)

                context_start = cleaned_text.rfind('.', 0, start_cleaned) + 1
                context_end = cleaned_text.find('.', end_cleaned)
                if context_end == -1:
                    context_end = len(cleaned_text)
                context_sentence = cleaned_text[context_start:context_end + 1].strip()

                if project_name.lower() not in context_sentence.lower():
                    continue

                structured_data.append({
                    "pdf_file": pdf_file,
                    "page_number": page_number,
                    "project_name": project_name,
                    "context_sentence": context_sentence,
                    "coordinates": None
                })

            return structured_data
        except Exception as e:
            raise MyException(f"Error in extracting structured PROJECT entities: {e}")
        
    # Do the all above things in a pipeline 
    def predict(self) -> List[Dict[str, Any]]:
        """Main driver method to extract PROJECT entities from PDF."""
        try:
            all_structured_data = []
            page_texts = self.extract_text_per_page()

            for page_data in page_texts:
                text = page_data["text"]
                pdf_file = page_data["pdf_file"]
                page_number = page_data["page_number"]

                entities = self.prediction(text)
                structured_data_for_page = self.extract_structured_jsonl(
                    entities, text, pdf_file, page_number
                )

                # De-duplication
                seen_projects = set()
                unique_data = []
                for entry in structured_data_for_page:
                    pname = entry["project_name"]
                    if pname not in seen_projects:
                        unique_data.append(entry)
                        seen_projects.add(pname)

                all_structured_data.extend(unique_data)

            return all_structured_data
        except MyException:
            raise
        except Exception as e:
            raise MyException(f"Unhandled exception in PDF processing: {e}")
