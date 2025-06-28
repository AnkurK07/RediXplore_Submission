# prompt_templates.py

llm_prompt = """
You are an expert data-cleaning assistant for mining project metadata.

You will receive a JSON list of project entries. Each entry contains:
- 'pdf_file'
- 'page_number'
- 'project_name'
- 'context_sentence'

Your task is to clean this data carefully in steps.

Remove any entry where 'project_name' is incomplete, fragmentary, or nonsensical. For example, remove names that look cut off or meaningless.

Compare entries for semantic similarity of 'project_name'. If two names refer to the same project with slight wording variations check word variation carefully character by character, keep only one.

Pick the most complete or most informative form of the name.

After duplication removal, analyze each project  in detail to determine whether it is a valid project name or not. If it is valid, keep it; if not, remove it.

If you think based on the 'context_sentence' the name of the project should be modified, do it.

Produce your final output as a valid JSON array containing only the cleaned and deduplicated entries. Do not include any explanations, analysis, markdown, or extra text. The entire output must be a single JSON array only.

Example desired output format:

[
  {{
    "pdf_file": "Report.pdf",
    "page_number": 1,
    "project_name": "Mangaroon Gold Project",
    "context_sentence": "..."
  }}
]

Now clean this data:

{all_projects}
"""
