from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from config import db, model


def retrieve_documents(query, k=5):

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20
        }
    )

    docs = retriever.invoke(query)

    return docs


def generate_answer(query, docs):

    combined_input = f"""
Answer the question using ONLY the
provided documents.

Question:
{query}

Documents:
{chr(10).join([doc.page_content for doc in docs])}

If the answer cannot be found in the
documents, say:

'I do not have enough information from
the provided documents.'
"""

    messages = [

        SystemMessage(
            content="""
                    You are a legal assistant.

                    Rules:

                    1. Answer strictly from provided context.
                    2. Never use outside knowledge.
                    3. Never hallucinate.
                    4. If information is missing say:

                    'I do not have enough information from
                    the provided documents.'
                    """
        ),

        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)

    return result.content


def ask_rag(query):

    docs = retrieve_documents(query)

    print("\n--- Retrieved Context ---")

    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}")
        print(doc.page_content[:500])

    answer = generate_answer(query, docs)

    return answer