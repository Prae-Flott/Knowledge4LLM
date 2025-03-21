import ollama
from utils.text_spliter import parse_file
from utils.embedding_utils import get_embeddings, find_most_similar

def main():
    SYSTEM_PROMPT = (
        "You are a helpful reading assistant who answers questions based on snippets of text provided in context. "
        "Answer only using the context provided, being as concise as possible. If you're unsure, just say that you don't know.\n"
        "Context:\n"
    )
    # Read and split the text file into paragraphs.
    filename = "peter-pan.txt"
    paragraphs = parse_file(filename)

    # Generate or load embeddings for the paragraphs.
    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)

    # Get a query from the user and generate its embedding.
    prompt = input("What do you want to know? -> ")
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]

    # Find the top 5 most similar paragraphs based on cosine similarity.
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    # Create a chat prompt by combining a system prompt and the context from the similar paragraphs.
    response = ollama.chat(
        model="deepseek-r1:14b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                           + "\n".join(paragraphs[idx] for _, idx in most_similar_chunks),
            },
            {"role": "user", "content": prompt},   
        ],
    )
    print("\n\n")
    print(response["message"]["content"])

if __name__ == "__main__":
    main()
