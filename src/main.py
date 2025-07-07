import warnings

def filter_langchain_warnings(message, category, filename, lineno, file=None, line=None):
    if "LangChainDeprecationWarning" in str(message):
        return
    return warnings.defaultaction

warnings.showwarning = filter_langchain_warnings

from kg_qa_chain import create_kg_qa_chain
def main():
    print("Fast & Furious Knowledge Graph QA (Neo4j + LangChain + Ollama)\n")
    print("Digite sua pergunta sobre o universo Velozes e Furiosos (ou 'sair' para encerrar):\n")

    chain = create_kg_qa_chain()

    while True:
        question = input("Pergunta: ").strip()
        if question.lower() in ["sair", "exit", "quit"]:
            print("Encerrando...")
            break

        if not question:
            print("Pergunta vazia. Tente novamente.")
            continue

        try:
            result = chain.invoke({"query": question})
            print(f"Resposta: {result['result']}\n")
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}\n")

if __name__ == "__main__":
    main()
