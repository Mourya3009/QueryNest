from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

CHROMA_PATH = "./chroma_db"


def get_embeddings():
    # nomic-embed-text runs via Ollama locally
    # Same model used for both indexing and querying (critical!)
    return OllamaEmbeddings(model="nomic-embed-text")


def get_vectorstore():
    # persist_directory tells ChromaDB where to save data
    # Data survives application restarts
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )


def ingest_document(file_path: str) -> int:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # ~375 words per chunk
        chunk_overlap=50,   # 50 token overlap between chunks
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def retrieve_relevant_chunks(query: str, k: int = 3) -> str:
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    if not results:
        return "No relevant information found in the uploaded documents."

    return "\n\n".join([
        f"[Source chunk {i+1}]:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])
