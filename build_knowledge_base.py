from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME


def load_documents():
    documents = []
    for md_file in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": md_file.name}))
    return documents


def main():
    print("Loading policy documents...")
    documents = load_documents()
    print(f"Found {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_STORE_DIR))
    print(f"Done. Saved index to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
