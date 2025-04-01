import ollama
from utils.text_spliter import parse_file
from utils.embedding_utils import get_knowledge_context
from data_loader import read_latest_description


def main():
    SYSTEM_PROMPT = (
        "You are a helpful reading assistant who answers questions based on snippets of text provided in context. "
        "Answer only using the context provided, being as concise as possible. If you're unsure, just say that you don't know.\n"
        "Context:\n\n"
    )

    # Get user query
    user_query = input("What would you like to know? ")

    # Retrieve relevant context from knowledge base
    context, sources = get_knowledge_context(
        user_query, 
        knowledge_dir="./knowledge_base", 
        modelname="nomic-embed-text", 
        top_k=10)

    # Create a chat prompt by combining a system prompt and the context from the similar paragraphs.
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + context,
            },
            {"role": "user", "content": user_query},   
        ],
    )
    
    # Display response
    print("\nAnswer:")
    print(response["message"]["content"])
    if sources:
        print("\nRetrieving information from:")
        for source in sources:
            print(f"- {source}")
    else:
        print("No relevant information found in knowledge base.")
        return

if __name__ == "__main__":
    main()