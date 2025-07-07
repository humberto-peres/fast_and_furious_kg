from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.chat_models import ChatOllama
import config

def create_kg_qa_chain():
    graph = Neo4jGraph(
        url=config.NEO4J_URL,
        username=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        system=config.KG_SYSTEM_PROMPT
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    return chain
