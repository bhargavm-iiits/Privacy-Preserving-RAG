from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

class PIIAnonymizer:
    def __init__(self):
        print("Loading Microsoft Presidio NLP Models (this takes a few seconds)...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Add a pattern recognizer for US SSN
        ssn_pattern = Pattern(name="US_SSN", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.8)
        ssn_recognizer = PatternRecognizer(supported_entity="US_SSN", patterns=[ssn_pattern])
        self.analyzer.registry.add_recognizer(ssn_recognizer)

        # Add a pattern recognizer for hospital/organization names (very basic)
        hospital_pattern = Pattern(name="HOSPITAL", regex=r"[A-Z][a-z]+ (General )?Hospital", score=0.5)
        hospital_recognizer = PatternRecognizer(supported_entity="ORGANIZATION", patterns=[hospital_pattern])
        self.analyzer.registry.add_recognizer(hospital_recognizer)

    def anonymize_chunk(self, text: str) -> str:
        """Scans the text for PII and replaces it with generic tags."""
        # 1. Analyze: Find where the sensitive data is hiding
        analysis_results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=["PERSON", "LOCATION", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "ORGANIZATION"]
        )
        # 2. Anonymize: Swap the sensitive data for tags like <ENTITY_TYPE>
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analysis_results # type: ignore
        )
        return anonymized_result.text

# ==========================================
# Let's test it on your exact terminal output!
# ==========================================
if __name__ == "__main__":
    redactor = PIIAnonymizer()

    # The exact chunk your PreProcessing.py script just generated
    my_chunk = "SUBJECTIVE:, This 23-year-old white female presents with complaint of allergies. She used to have allergies when she lived in Seattle but she thinks they are worse here. In the past, she has tried Claritin, and Zyrtec. Both worked for short time but then seemed to lose effectiveness. She has used Allegra also. She used that last summer and she began using it again two weeks ago. It does not appear to be working very well. She has used over-the-counter sprays but no prescription nasal sprays. She"

    print("\n--- ORIGINAL CHUNK ---")
    print(my_chunk)

    safe_chunk = redactor.anonymize_chunk(my_chunk)

    print("\n--- ANONYMIZED CHUNK ---")
    print(safe_chunk)