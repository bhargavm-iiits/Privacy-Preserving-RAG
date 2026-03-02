import chromadb
from langchain_chroma import Chroma
from DP_Embeddings import DifferentiallyPrivateEmbedder
from langchain_core.documents import Document

# 1. Initialize our mathematical shield (the DP Embedder we just built)
# We use a slightly higher epsilon here (e.g., 2.0) so the AI can still accurately find the documents.
dp_embedder = DifferentiallyPrivateEmbedder(epsilon=2.0)

# 2. Setup ChromaDB Local Storage
# This creates a folder called 'chroma_db' in your project directory to save the data permanently
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection (think of this like a table in a SQL database)
collection_name = "secure_medical_records"

# 3. Connect LangChain to ChromaDB using our custom DP Embedder
vector_store = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=dp_embedder, # type: ignore
)

print("✅ Secure ChromaDB Initialized!")

# ==========================================
# Let's test adding a document to the vault!
# ==========================================
if __name__ == "__main__":
    
    # Imagine this is the exact output from your Master_Pipeline.py
    safe_text = "SUBJECTIVE:, This <DATE_TIME_4> white female presents with complaint of allergies. She used to have allergies when she lived in <LOCATION_3> but she thinks they are worse here."
    
    metadata = {
        "record_id": "MT_0", 
        "doc_type": "Transcription", 
        "chunk_index": 0
    }
    
    # Create the LangChain document
    doc_to_store = Document(page_content=safe_text, metadata=metadata)
    
    print("\nEncrypting and storing document into ChromaDB...")
    
    # Add the document to the database
    # Under the hood, this automatically runs our DP_Embeddings logic!
    vector_store.add_documents([doc_to_store])
    
    print("🔒 Document safely stored with Differential Privacy!")
    
    # 4. Let's test searching the database!
    print("\n--- TESTING RETRIEVAL ---")
    query = "Where did the patient live?"
    
    print(f"Doctor asks: '{query}'")
    # This embeds the query, adds DP noise, and finds the closest matching vector
    results = vector_store.similarity_search(query, k=1)
    
    if results:
        print("\nAI Retrieved Context:")
        print(f"Text: {results[0].page_content}")
        print(f"Metadata: {results[0].metadata}")