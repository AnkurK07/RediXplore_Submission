
# Import important libraries
from transformers import AutoModelForTokenClassification, AutoTokenizer
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cerebras import ChatCerebras
from langchain import hub

from NER.Prediction import ModelPrediction
from Intelligent_Agent.agent_executor import AgentResult


# Import finetuned model from my huggingface account and save it to your local and then load and run this.
model = AutoModelForTokenClassification.from_pretrained("kankur0007/BERT-NER-Projects",num_labels=9, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained("kankur0007/BERT-NER-Projects")



# Pick a llm. Here i'm taking a free opensource llm api , you can choose your own.

llm = ChatCerebras(
    model="llama-3.3-70b",
    api_key="API_KEY",
)


 # Pick a search tool for AI Agent that can search lat,long again i'm choosing a free opensource for experiment.
 # Lat,long search can be depend on search tool you can use your own paid tool.

TAVILY_API_KEY = 'API_KEY'
search = TavilySearchAPIWrapper(tavily_api_key = TAVILY_API_KEY)

search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

tools = [
    Tool(
        name="Tavily Search",
        func=search_tool.run, # Use the run method of the TavilySearchResults tool
        description="Useful for searching the web to find up-to-date information about project locations or details."
    )
]




# Making a AI Agent that can search lat,long via tool calling.
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Now predict the result with model
mp = ModelPrediction(pdf_path='Report_5.pdf',model=model,tokenizer =tokenizer)
result = mp.predict() 

# Now get your coordinate(lat,long) by ai agent.
ar = AgentResult(data=result,llm=llm,agent_executor=agent_executor)
agent_result =  ar.result()
