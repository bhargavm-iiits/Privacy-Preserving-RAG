from typing import TypedDict
from langgraph.graph import StateGraph, END
import re

# 1. Define the "State" (The Clipboard our manager carries around)
class RAGState(TypedDict):
    question: str
    retrieved_context: str
    anonymized_answer: str
    final_clean_answer: str
    decoder_ring: dict  # This is crucial! It holds our secrets.

# 2. Define the Nodes (The specific tasks the manager performs)

def retrieve_node(state: RAGState):
    print("🕵️‍♂️ Manager: Searching the secure ChromaDB vault...")
    # In reality, this connects to your Secure_Store.py. 
    # For this test, we simulate what ChromaDB just returned to us:
    retrieved_text = "The patient is a <DATE_TIME_4> female. She lived in <LOCATION_3> and her allergies are worse there."
    return {"retrieved_context": retrieved_text}

def generate_node(state: RAGState):
    print("🤖 Manager: Asking the AI to generate an answer...")
    # Here is where you would normally call your LLM (e.g., Llama 3 or GPT-4)
    # The AI reads the context and writes a helpful sentence, but it ONLY knows the safe tags!
    mock_llm_output = "Based on the records, the patient lived in <LOCATION_3>. They are a <DATE_TIME_4> female."
    return {"anonymized_answer": mock_llm_output}

def deanonymize_node(state: RAGState):
    print("🔏 Manager: Intercepting answer and restoring real names using the Decoder Ring...")
    answer = state["anonymized_answer"]
    ring = state["decoder_ring"]
    
    # Swap the tags back to the real words
    for tag, real_word in ring.items():
        answer = answer.replace(tag, real_word)
        
    return {"final_clean_answer": answer}

# 3. Build the LangGraph Workflow
workflow = StateGraph(RAGState)

# Add our nodes
workflow.add_node("Retrieve", retrieve_node)
workflow.add_node("Generate", generate_node)
workflow.add_node("De_Anonymize", deanonymize_node)

# Connect them in order
workflow.set_entry_point("Retrieve")
workflow.add_edge("Retrieve", "Generate")
workflow.add_edge("Generate", "De_Anonymize")
workflow.add_edge("De_Anonymize", END)

# Compile the brain!
app = workflow.compile()

# ==========================================
# Let's test the entire pipeline end-to-end!
# ==========================================
if __name__ == "__main__":
    print("\n--- NEW DOCTOR QUERY ---")
    
    # We load the decoder ring from our Master_Pipeline step
    my_decoder_ring = {
        "<LOCATION_3>": "Seattle",
        "<DATE_TIME_4>": "23-year-old"
    }
    
    # The starting clipboard
    initial_state = {
        "question": "Where did the patient live and how old are they?",
        "decoder_ring": my_decoder_ring
    }
    
    # Run the manager!
    result = app.invoke(initial_state) # type: ignore
    
    print("\n--- FINAL OUTPUT TO DOCTOR ---")
    print(f"Doctor asked: {result['question']}")
    print(f"AI actually generated: {result['anonymized_answer']}")
    print(f"Manager successfully converted it to: {result['final_clean_answer']}\n")