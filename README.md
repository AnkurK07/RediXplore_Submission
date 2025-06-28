## 📍 RediXplore: Intelligent Project Extraction & Geolocation Pipeline
This project is designed to automatically extract mining project names from unstructured PDF documents, and enrich each identified project with its geographic coordinates (latitude and longitude). It uses a domain-specific NER model to extract project entities from the text, then leverages a web-augmented intelligent agent to determine the most accurate location for each project.

### 📌 Model & Geolocation Strategy
![image](https://github.com/user-attachments/assets/79fe3925-a87f-4c67-bc84-e3e41b41e382)

📄 The input is an unstructured PDF, typically a mining report or technical document. This file is fed into the PDF Preprocessor, which extracts raw text page-by-page and cleans it, removing, formatting noise, fixing spacing, and normalizing characters so it's ready for structured analysis.

🧹 The cleaned text is then passed to the NER Model, a fine-tuned bert-base-cased model using PEFT (LoRA). This model is trained to recognize and extract named entities, particularly mining PROJECT names. <br>

**⚠️ However, due to overfitting, the model tends to over predict marking nearly every word that appears next to "project" as a project name, which introduces a high number of false positives.**

**🧠 To resolve this, the extracted raw project names are passed through an LLM Validation module, powered by LLaMA 3.3 70B (via Cerebras API). This large language model is capable of understanding context and semantics. It performs:**

- Validation – removing fragmented, meaningless, or cut-off project names.

- Deduplication – identifying and merging near-duplicate names using character and meaning level comparison.

- Correction – updating the name if the surrounding context suggests a more accurate or complete form.

🌍 The validated project names are then enriched using an AI Agent, which is orchestrated via LangChain and Tavily web search. This agent attempts to find accurate geographic coordinates for each project name by combining document context with real-time web data. The result is (lat,lon) information for each project.

📦 Finally, the pipeline compiles all enriched project entries—including pdf_file, page_number, project_name, context_sentence, and coordinates—into a clean JSON output. This structured format is ready for downstream geospatial analysis, visualization, or knowledge graph integration.

### 🛠️ Setup & Execution 

#### 1. Clone the Repository 
```
git clone https://github.com/AnkurK07/RediXplore_Submission.git
```
#### 2. Create a virtual environment and install requirements.txt
```
pip install -r requirements.txt
```

#### 3. Load finetuned model from huggingface , set up llms , search tools and run this python module.
```
python intelligent_pipeline.py
```

### 🧰 Tools, Libraries, and APIs Used

| **Component**              | **Technology / Tool**                                              | **Purpose**                                                               |
|----------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------|
| **NER Model**              | `bert-base-cased` (fine-tuned with **PEFT LoRA**)                  | Custom entity recognition for mining project names                        |
|                            | `transformers`, `peft`, `datasets`, `evaluate`, `seqeval`          | Model loading, fine-tuning, evaluation                                   |
| **Model Hosting**          | `Hugging Face Transformers`                                        | Load and serve the fine-tuned model                                       |
| **PDF Text Extraction**    | `PyMuPDF` (`fitz`)                                                 | Extracts text from PDFs page-by-page                                      |
| **Text Preprocessing**     | `re`, `AutoTokenizer`                                              | Cleans and tokenizes input text                                           |
| **Prompt Agent Pipeline**  | `langchain`, `langchain-cerebras`, `langchain-tavily`              | Agent orchestration, search-enabled reasoning                             |
| **LLM for Geolocation**    | `LLaMA 3.3 70B` via `Cerebras API`                                 | Finds accurate coordinates using web-based search                         |
| **Web Search API**         | `Tavily`                                                           | Supplies real-time information to the LLM agent                           |
| **Coordinate Parsing**     | Custom regex logic (`extract_coordinates`)                         | Converts various lat/lon formats into clean decimal format                |





