from langchain_openai import AzureChatOpenAI
from config import AZURE
 
def get_llms():
    # Regular LLM for precise, document-grounded QA queries
    normal_llm = AzureChatOpenAI(
        azure_endpoint=AZURE["endpoint"],
        api_key=AZURE["api_key"],
        api_version=AZURE["api_version"],
        azure_deployment=AZURE["chat"],
        temperature=0.3
    )
 
    # Creative LLM used primarily when handling alternative retries or brainstorming
    creative_llm = AzureChatOpenAI(
        azure_endpoint=AZURE["endpoint"],
        api_key=AZURE["api_key"],
        api_version=AZURE["api_version"],
        azure_deployment=AZURE["chat"],
        temperature=0.9
    )
 
    return normal_llm, creative_llm