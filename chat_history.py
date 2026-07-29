from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from config import model
from retrieval_pipeline import ask_rag

chat_history = []


def rewrite_question(user_question):

    if not chat_history:
        return user_question

    messages = [

        SystemMessage(
            content="""
            Rewrite the new question so it becomes
            standalone and searchable.

            Return ONLY the rewritten question.
            """
        )

    ] + chat_history + [

        HumanMessage(
            content=f"New question: {user_question}"
        )
    ]

    result = model.invoke(messages)

    return result.content.strip()


def ask_question(user_question):

    print(f"\nYou asked: {user_question}")

    search_question = rewrite_question(
        user_question
    )

    print(
        f"\nSearching for:\n{search_question}"
    )
     

    answer = ask_rag(search_question)

    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    print("\n--- Answer ---\n")
    print(answer)


def start_chat():

    print(
        "\nLegal RAG Assistant\n"
        "Type 'quit' to exit.\n"
    )

    while True:

        question = input("Your question: ")
        print(repr(question))

        if question.lower() == "quit":
            print("Goodbye.")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()