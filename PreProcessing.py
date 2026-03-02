import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Let's reuse our Preprocessor from the previous step
class MedicalDataPreprocessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )

    def clean_text(self, text):
        import re
        if not isinstance(text, str): return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def process_record(self, record_id, raw_text, doc_type, extra_metadata=None):
        cleaned_text = self.clean_text(raw_text)
        if not cleaned_text: return []
        
        chunks = self.text_splitter.create_documents([cleaned_text])
        for i, chunk in enumerate(chunks):
            # Base metadata
            meta = {"record_id": str(record_id), "doc_type": doc_type, "chunk_index": i}
            # Add any extra metadata (like patient ID or department)
            if extra_metadata: meta.update(extra_metadata)
            chunk.metadata = meta
            
        return chunks

# ==========================================
# How to Load YOUR Specific Datasets
# ==========================================
preprocessor = MedicalDataPreprocessor(chunk_size=500, chunk_overlap=50)
all_processed_chunks = []

# --- 1. Load MTSamples (Level 2 Test) ---
print("Loading MTSamples...")
mtsamples_path = r"C:\Users\Bhargav M\My_Projects\Gen_AI\DataSets\Medical\mtsamples.csv"
df_mt = pd.read_csv(mtsamples_path)

# Let's just process the first 5 rows to test
for index, row in df_mt.head(5).iterrows():
    # MTSamples usually has 'medical_specialty' and 'transcription' columns
    chunks = preprocessor.process_record(
        record_id=f"MT_{index}",
        raw_text=row.get('transcription', ''),
        doc_type="Transcription",
        extra_metadata={"specialty": row.get('medical_specialty', 'Unknown')}
    )
    all_processed_chunks.extend(chunks)

# --- 2. Load MIMIC-III Note Events (Level 3 Test) ---
print("Loading MIMIC-III Notes...")
mimic_notes_path = r"C:\Users\Bhargav M\My_Projects\Gen_AI\DataSets\MIMIC\mimic-iii-clinical-database-demo-1.4\NOTEEVENTS.csv"

# NOTEEVENTS can be huge, even in the demo. We use nrows=5 to just grab the first 5 notes.
df_mimic = pd.read_csv(mimic_notes_path, nrows=5) 

for index, row in df_mimic.iterrows():
    # MIMIC NOTEEVENTS has 'ROW_ID', 'CATEGORY', and 'TEXT' columns
    chunks = preprocessor.process_record(
        record_id=f"MIMIC_{row.get('ROW_ID', index)}",
        raw_text=row.get('TEXT', ''),
        doc_type=row.get('CATEGORY', 'Clinical_Note'),
        extra_metadata={"subject_id": str(row.get('SUBJECT_ID', ''))}
    )
    all_processed_chunks.extend(chunks)

print(f"\nSuccessfully created {len(all_processed_chunks)} total chunks ready for Anonymization!")

# Preview one chunk
print("\n--- Sample Chunk ---")
print(f"Content: {all_processed_chunks[0].page_content}")
print(f"Metadata: {all_processed_chunks[0].metadata}")