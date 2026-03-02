import pandas as pd
from Anonymization import PIIAnonymizer
from PreProcessing import MedicalDataPreprocessor

# 1. Initialize our tools
preprocessor = MedicalDataPreprocessor(chunk_size=500, chunk_overlap=50)
redactor = PIIAnonymizer()

# 2. We will store our LangGraph "Decoder Ring" (State) here
# Example: {"<LOCATION_1>": "Seattle"}
decoder_ring = {}
entity_counter = 1

print("\nLoading and Preprocessing MTSamples Data...")
mtsamples_path = r"C:\Users\Bhargav M\My_Projects\Gen_AI\DataSets\Medical\mtsamples.csv"
df_mt = pd.read_csv(mtsamples_path)

# Process just the first 2 rows for this test
all_chunks = []
for index, row in df_mt.head(2).iterrows():
    chunks = preprocessor.process_record(
        record_id=f"MT_{index}",
        raw_text=row.get('transcription', ''),
        doc_type="Transcription"
    )
    all_chunks.extend(chunks)

print(f"Generated {len(all_chunks)} chunks. Starting Anonymization...\n")

# 3. Anonymize and build the Decoder Ring
anonymized_database = []

for chunk in all_chunks:
    original_text = chunk.page_content
    
    # Run Presidio analysis
    analysis_results = redactor.analyzer.analyze(
        text=original_text,
        language='en',
        entities=["PERSON", "LOCATION", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS"]
    )
    
    # We have to reverse the results so we don't mess up the string indexes as we replace text
    analysis_results.sort(key=lambda x: x.start, reverse=True)
    
    anonymized_text = original_text
    
    # Manually replace text so we can save exactly what we replaced
    for result in analysis_results:
        original_word = original_text[result.start:result.end]
        
        # Create a unique tag like <LOCATION_1>, <LOCATION_2>
        unique_tag = f"<{result.entity_type}_{entity_counter}>"
        
        # Save it to our decoder ring
        decoder_ring[unique_tag] = original_word
        
        # Swap the word in the text
        anonymized_text = anonymized_text[:result.start] + unique_tag + anonymized_text[result.end:]
        
        entity_counter += 1
        
    # Update the chunk with the safe text
    chunk.page_content = anonymized_text
    anonymized_database.append(chunk)

# ==========================================
# Display the Results!
# ==========================================
print("--- FIRST ANONYMIZED CHUNK ---")
print(anonymized_database[0].page_content)

print("\n--- THE DECODER RING (Keep this secure!) ---")
for tag, real_word in list(decoder_ring.items())[:5]: # Show first 5
    print(f"{tag} = {real_word}")