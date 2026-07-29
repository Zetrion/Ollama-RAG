import warnings
warnings.filterwarnings("ignore")

import os

from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path):

    print(f"\nLoading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"{docs_path} does not exist."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No PDF files found in {docs_path}."
        )

    print(f"Loaded {len(documents)} pages.")

    return documents


def split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=0
):
    """
    Split documents into chunks.
    """

    print("\nSplitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vector_store(
        chunks,
        persist_directory
):
    """
    Create Chroma database.
    """

    print("\nCreating vector database...")

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434"
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    batch_size = 50

    total_batches = (
        len(chunks) + batch_size - 1
    ) // batch_size

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        print(
            f"Processing batch "
            f"{i//batch_size + 1}/{total_batches}"
        )

        vectorstore.add_documents(batch)

    print("\nFinished creating vector database.")

    return vectorstore


def main():

    # Ask user which knowledge base to build.

    knowledge_base = input(
        "Knowledge Base Name (Example: Kerala_Laws, AI, Medical): "
    ).strip()

    # Documents are now loaded from
    # Documents/<KnowledgeBase>

    docs_path = os.path.join(
        "Documents",
        knowledge_base
    )

    # Database is now stored in
    # db/<KnowledgeBase>

    persist_directory = os.path.join(
        "db",
        knowledge_base
    )

    documents = load_documents(docs_path)

    chunks = split_documents(documents)

    create_vector_store(
        chunks,
        persist_directory
    )

    print("\n====================================")
    print("Knowledge Base Created Successfully")
    print(f"Name : {knowledge_base}")
    print(f"Docs : {docs_path}")
    print(f"DB   : {persist_directory}")
    print("====================================")


if __name__ == "__main__":
    main()