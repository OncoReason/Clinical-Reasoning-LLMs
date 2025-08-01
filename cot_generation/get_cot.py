import requests
import json
import time
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
MAX_PROCESSES = 10
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PROMPT_TEMPLATE = """ """ # Note: The prompt templates will be made publicly available upon paper acceptance.

# Functions
def load_cot_data(file_path):
    """Load chain-of-thought data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_cot_data(file_path, data):
    """Save chain-of-thought data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_clinical_reasoning(input_text, retries=3):
    """
    Sends a request to the OpenRouter API to get clinical reasoning based on the input text.
    
    Args:
        input_text (str): The input text to send to the API.
        retries (int): Number of retries in case of failure (default is 3).
    
    Returns:
        str: The response content from the API.
    """
    system_prompt = () # Note: The prompts will be made publicly available upon paper acceptance.

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    for attempt in range(retries):
        try:
            # response = requests.post(url, headers=headers, data=json.dumps(data))
            # response.raise_for_status()  # Raise an exception for HTTP errors
            # response_data = response.json()
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                    "role": "user",
                    "content": input_text
                    }
                ],
                stream=False
            )
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                raise RuntimeError("Failed to get a response after multiple retries.") from e

def process_patient(patient):
    """Process a single patient's data."""
    patient_id = patient["patient_id"]
    input_text = PROMPT_TEMPLATE.format(
        patient_data=patient["patient_data"],
        survival_status=patient["survival_status"],
        survival_months=patient["survival_months"]
    )
    reasoning = get_clinical_reasoning(input_text)
    return patient_id, reasoning

# Main Execution
def main():
    try:
        cot_data = load_cot_data("patient_summary_prompt_data.json")
        print("Loaded cot_data.json successfully.")
    except FileNotFoundError:
        print("new_cot_data.json not found.")
        return
    
    # Attempt to load partial data if it exists
    try:
        processed_data = load_cot_data("cot_sorted.json")
        print("Loaded cot_data_partial.json successfully. Resuming from partial data.")
    except FileNotFoundError:
        print("cot_data_partial.json not found. Starting fresh.")
        processed_data = {}

    with ThreadPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        future_to_patient = {}
        for patient in cot_data:
            patient_id = patient["patient_id"]
            if patient_id in processed_data:
                print(f"Skipping patient {patient_id} as it is already processed.")
                continue
            future_to_patient[executor.submit(process_patient, patient)] = patient

        for future in as_completed(future_to_patient):
            patient = future_to_patient[future]
            try:
                patient_id, reasoning = future.result()
                if reasoning:
                    reasoning = reasoning.choices[0].message.content
                    processed_data[patient_id] = reasoning
                    print(f"Patient {patient_id} processed successfully.")
                else:
                    print(f"Failed to process patient {patient_id}.")
            except Exception as e:
                print(f"Error processing patient {patient['patient_id']}: {e}")
                print(reasoning)

            # Save partial results every 10 patients
            if len(processed_data) % 10 == 0:
                save_cot_data("cot_sorted.json", processed_data)

    # Save final results
    save_cot_data("cot_data_response_final5.json", processed_data)
    save_cot_data("cot_sorted.json", processed_data)

if __name__ == "__main__":
    main()
