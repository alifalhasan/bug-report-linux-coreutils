import pandas as pd
import ollama

# Load CSV file
csv_file = "../dataset/descriptions.csv"  # Change to your CSV filename
df = pd.read_csv(csv_file)

# Define the Ollama model (e.g., llama2, mistral, etc.)
model = "llama3.3:70b"

# Function to generate response
def generate_response(prompt):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Iterate through CSV and generate responses
responses = []
for index, row in df.iterrows():
    description = row['description']
    prompt = f"""
I will provide you with a Linux coreutils bug description that was reported by a developer in Bugzilla. Your task is to determine the exact command(s) or test case(s) required to reproduce the bug.

If a reproducible command or test case exists, write only the command or test case.

If no reproducible command or test case is available, write None.

Example:
Input (Bug Description):
A user reports that the ls command incorrectly sorts filenames when using the -v option. The issue occurs when filenames contain both numbers and letters, leading to unexpected sorting behavior.
touch file1 file2 file10 file20
ls -v


Output (Expected Response):
touch file1 file2 file10 file20
ls -v


Here is the reported Bug Description:

{description}

Task: Given the above bug description, identify the commands or test cases required to reproduce the bug.
"""
    response = generate_response(prompt)
    responses.append(response)
    print(response)
    print("------------------------------")

# Add responses to DataFrame and save
df["Ollama_Response"] = responses
df.to_csv("manual-deepseek-prompt-v1.csv", index=False)

print("Processing complete. Results saved in processed_data.csv")
