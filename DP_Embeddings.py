import numpy as np
# Look here: We updated the import to the new standard!
from langchain_huggingface import HuggingFaceEmbeddings

class DifferentiallyPrivateEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", epsilon=1.0, sensitivity=1.0):
        print(f"Loading Local Embedding Model: {model_name}...")
        self.base_embedder = HuggingFaceEmbeddings(model_name=model_name)
        
        # Privacy Parameters
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
        # Calculate the standard deviation for the Gaussian noise
        # Lower epsilon = higher scale = more noise
        self.noise_scale = self.sensitivity / self.epsilon

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query with DP noise."""
        raw_vector = self.base_embedder.embed_query(text)
        return self._add_noise(raw_vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents with DP noise."""
        raw_vectors = self.base_embedder.embed_documents(texts)
        return [self._add_noise(vec) for vec in raw_vectors]

    def _add_noise(self, vector: list[float]) -> list[float]:
        """The mathematical shield: Injects Gaussian noise into the vector array."""
        vector_array = np.array(vector)
        
        # Generate random noise based on our privacy budget (epsilon)
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=vector_array.shape)
        
        # Add the noise to the original vector
        dp_vector = vector_array + noise
        
        return dp_vector.tolist()

# ==========================================
# Let's test it on your anonymized chunk!
# ==========================================
if __name__ == "__main__":
    # We will set Epsilon to 0.5 (A moderate privacy budget)
    dp_embedder = DifferentiallyPrivateEmbedder(epsilon=0.5)

    # Let's use the exact output from your last script
    anonymized_text = "SUBJECTIVE:, This <DATE_TIME_4> white female presents with complaint of allergies. She used to have allergies when she lived in <LOCATION_3> but she thinks they are worse here."

    print("\nGenerating Standard (Unsafe) Embedding...")
    unsafe_vector = dp_embedder.base_embedder.embed_query(anonymized_text)
    
    print("Generating Differentially Private (Safe) Embedding...")
    safe_vector = dp_embedder.embed_query(anonymized_text)

    # A vector is a list of 384 numbers. Let's just look at the first 5 to see the difference.
    print("\n--- COMPARING THE VECTORS (First 5 dimensions) ---")
    print(f"Original: {unsafe_vector[:5]}")
    print(f"DP Noisy: {safe_vector[:5]}")