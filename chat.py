from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load vector DB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Load Phi-3 model
llm = Ollama(model="phi3")

print("\nRAG Chat Started (type 'exit' to quit)\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer strictly from the provided context.
    Do not add extra information.
    If the answer is not in the context, say "Not found in document".
    Be concise.

    Context:
    {context}

    Question:
    {query}
    """

    answer = llm.invoke(prompt)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for doc in docs:
        print(doc.metadata)

    print("\n" + "-" * 50)