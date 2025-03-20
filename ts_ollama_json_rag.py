import ollama
from utils.json_spliter import split_data_by_count
from utils.embedding_utils import get_embeddings, find_most_similar

def main():
    SYSTEM_PROMPT = (
        "You are a data analysis expert specializing in time series. Analyze the provided time series data (timestamps and numerical measurements).\n"
        "Identify trends, anomalies, and key statistics, and explain your findings clearly and concisely based solely on the data.\n"
        "Answer only using the context provided, being as concise as possible. If you're unsure, just say that you don't know.\n"
        "Context:\n"
    )
    # Read and split the text file into paragraphs.
    filename = "battery_status.json"
    data_chunks = split_data_by_count(filename, 10)

    # Generate or load embeddings for the paragraphs.
    embeddings = get_embeddings(filename, "nomic-embed-text", data_chunks)

    # Get a query from the user and generate its embedding.
    prompt = input("What do you want to know? -> ")
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]

    # Find the top 5 most similar paragraphs based on cosine similarity.
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    # Create a chat prompt by combining a system prompt and the context from the similar paragraphs.
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                           + "\n".join(data_chunks[idx] for _, idx in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\n\n")
    print(response["message"]["content"])

if __name__ == "__main__":
    main()
