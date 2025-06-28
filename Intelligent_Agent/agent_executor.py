# agent_result.py
import json
from typing import List, Dict
from exception import MyException


from prompt_template import llm_prompt
from Utils.utility import extract_coordinates

class AgentResult:
    def __init__(self, data: List[Dict], llm, agent_executor):
        self.data = data
        self.llm = llm
        self.agent_executor = agent_executor


    # For LLM Validation of predicted project
    def validate_projects_bulk(self) -> List[Dict]:
        try:
            prompt = llm_prompt.format(all_projects=json.dumps(self.data, indent=2))
            response = self.llm.invoke(prompt)
            print(f"\n 🖨️ Here is your projects....:\n{response.content}")
            cleaned_projects = json.loads(response.content)
            return cleaned_projects
        except Exception as e:
            print(f"Bulk LLM validation failed: {e}")
            return []
        
    # 
    def result(self) -> List[Dict]:
        enriched_output = []
        print("\n'🔍 Searching....... your projects'")
        cleaned_valid_projects = self.validate_projects_bulk()

        if not cleaned_valid_projects:
            print("❌ No valid projects found after validation.")
            return enriched_output

        print(f"✅ {len(cleaned_valid_projects)} valid projects identified.")

        for entry in cleaned_valid_projects:
            print(f"\n🌟 Processing: {entry['project_name']}")
            try:
                agent_input = {
                    "input": (
                        f"Find the most accurate geographic coordinates (latitude and longitude) for the mining or resource project named '{entry['project_name']}'. "
                        f"Context from document: '{entry['context_sentence']}'. "
                        f"If not in context, use web search. Return the result as '(latitude, longitude)' in decimal degrees, or 'None' if ambiguous/unknown."
                    )
                }
                agent_response = self.agent_executor.invoke(agent_input)
                print(f"🤖 Agent Output: {agent_response['output']}")
                coordinates = extract_coordinates(agent_response['output'])
                if coordinates is None:
                    print("⚠️ Agent returned no coordinates or invalid format.")
            except Exception as e:
                print(f"⚠️ Agent error: {e}")
                coordinates = None

            enriched_output.append({
                "pdf_file": entry["pdf_file"],
                "page_number": entry["page_number"],
                "project_name": entry["project_name"],
                "context_sentence": entry["context_sentence"],
                "coordinates (lat,long)": coordinates
            })

        return enriched_output
