import unittest
from Anonymization import PIIAnonymizer
from DP_Embeddings import DifferentiallyPrivateEmbedder

class TestPIILeakage(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """This runs once before the tests start."""
        print("\n--- Initializing CI/CD Security Audit ---")
        cls.redactor = PIIAnonymizer()
        cls.embedder = DifferentiallyPrivateEmbedder(epsilon=1.0)
        
        # A highly sensitive fake record
        cls.raw_sensitive_text = (
            "Patient John Doe, SSN 000-11-2222, DOB 05/12/1980, "
            "was treated at Seattle General Hospital. "
            "Call his wife Mary at 555-0198."
        )

    def test_anonymizer_scrubs_names(self):
        """Test 1: Does Presidio actually remove the names?"""
        safe_text = self.redactor.anonymize_chunk(self.raw_sensitive_text)
        
        # The test fails if "John Doe" or "Mary" is still in the text
        self.assertNotIn("John Doe", safe_text, "CRITICAL LEAK: Patient Name detected!")
        self.assertNotIn("Mary", safe_text, "CRITICAL LEAK: Family Name detected!")
        
    def test_anonymizer_scrubs_numbers(self):
        """Test 2: Does Presidio scrub SSNs and Phone Numbers?"""
        safe_text = self.redactor.anonymize_chunk(self.raw_sensitive_text)
        
        self.assertNotIn("000-11-2222", safe_text, "CRITICAL LEAK: SSN detected!")
        self.assertNotIn("555-0198", safe_text, "CRITICAL LEAK: Phone Number detected!")

    def test_embedding_pipeline_is_safe(self):
        """Test 3: Does the final math vector only receive scrubbed text?"""
        safe_text = self.redactor.anonymize_chunk(self.raw_sensitive_text)
        
        # We ensure that the text going into the embedder does NOT contain the hospital name
        self.assertNotIn("Seattle General Hospital", safe_text, "CRITICAL LEAK: Location going to DB!")
        
        # If it's safe, we ensure the math engine actually runs without crashing
        try:
            vector = self.embedder.embed_query(safe_text)
            self.assertEqual(len(vector), 384, "Embedding failed to generate 384 dimensions.")
        except Exception as e:
            self.fail(f"Embedding engine crashed on safe text: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)