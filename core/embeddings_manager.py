from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embeddings():
    # The specialized BGE model requested for the project
    model_name = "BAAI/bge-small-en-v1.5"
    
    print(f"🔄 Loading local BGE embedding model: {model_name}...")
    
    # BGE models expect specific parameters that this class handles natively
    model_kwargs = {'device': 'cpu'} # Change to 'mps' if on a Mac M1/M2/M3 for speed
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this sentence for searching relevant passages: "
    )
    
    return embeddings

if __name__ == "__main__":
    # Test script
    e = get_embeddings()
    print("✅ Embeddings loaded successfully!")