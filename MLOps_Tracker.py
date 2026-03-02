import mlflow
import random
import time

# --- THE FIX ---
# This explicitly creates a local database file named 'mlruns.db' in your current folder
# bypassing the Windows space-in-username bug completely.
mlflow.set_tracking_uri("sqlite:///mlruns.db")

# 1. Name your project in MLflow
mlflow.set_experiment("Privacy_Utility_Tradeoff")

def simulate_rag_accuracy(epsilon: float) -> float:
    """
    Simulates RAG retrieval accuracy. 
    Mathematically: Lower epsilon = More Noise = Lower Accuracy.
    """
    base_accuracy = 98.0
    noise_penalty = 10.0 / epsilon if epsilon > 0 else 60.0
    final_accuracy = base_accuracy - noise_penalty + random.uniform(-2, 2)
    return max(0.0, min(100.0, final_accuracy))

# 2. Define the Privacy Budgets we want to test
epsilons_to_test = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]

print("📊 Starting MLflow Experiment Runs...\n")

for eps in epsilons_to_test:
    # 3. Start a tracking run
    with mlflow.start_run(run_name=f"Epsilon_{eps}"):
        
        print(f"Testing RAG with Privacy Budget (Epsilon) = {eps}...")
        
        # Log our exact setting (Parameter)
        mlflow.log_param("privacy_budget_epsilon", eps)
        
        time.sleep(1)
        accuracy = simulate_rag_accuracy(eps)
        
        # Log the result (Metric)
        mlflow.log_metric("rag_retrieval_accuracy", accuracy)
        
        print(f"   ↳ Result Logged: Accuracy = {accuracy:.2f}%\n")

print("✅ All experiments logged successfully!")
print("👉 NEXT STEP: Type 'mlflow ui --backend-store-uri sqlite:///mlruns.db' in your terminal!")