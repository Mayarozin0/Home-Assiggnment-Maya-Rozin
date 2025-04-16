import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


def create_embedding(text):
    """Create an embedding vector for the given text using Azure OpenAI."""
    try:
        # Create embedding
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_DEPLOYMENT
        )

        # Extract the embedding from the response
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None


def load_json_file(file_path):
    """Load and read a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def prepare_text_for_embedding(json_data):
    """Convert JSON data to a text representation for embedding."""
    if not json_data:
        return ""

    # Create a structured text representation of the JSON data
    text_parts = []

    # Add category and description
    text_parts.append(f"Category: {json_data.get('category', '')}")
    text_parts.append(f"Description: {json_data.get('description', '')}")

    # Add HMO and tier information
    text_parts.append(f"HMO: {json_data.get('hmo', '')}")
    text_parts.append(f"Tier: {json_data.get('tier', '')}")

    # Add services information
    if 'services' in json_data and json_data['services']:
        text_parts.append("Services:")
        for service in json_data['services']:
            text_parts.append(f"  - Service Name: {service.get('name', '')}")
            text_parts.append(f"    Benefits: {service.get('benefits', '')}")

    # Add contact information
    if 'contact' in json_data and json_data['contact']:
        text_parts.append("Contact Information:")
        for key, value in json_data['contact'].items():
            text_parts.append(f"  - {key}: {value}")

    # Join all parts with newlines
    return "\n".join(text_parts)


def embed_knowledge_base(processed_data_dir):
    """Embed all JSON files in the knowledge base."""
    # Create a list to store embedding records
    embedding_records = []

    # Check if directory exists
    if not os.path.exists(processed_data_dir):
        print(f"Directory does not exist: {processed_data_dir}")
        return embedding_records

    # Counter for reporting
    files_found = 0
    files_embedded = 0

    # Traverse the directory structure
    for service_dir in os.listdir(processed_data_dir):
        service_path = os.path.join(processed_data_dir, service_dir)
        if os.path.isdir(service_path):
            for hmo_dir in os.listdir(service_path):
                hmo_path = os.path.join(service_path, hmo_dir)
                if os.path.isdir(hmo_path):
                    for tier_file in os.listdir(hmo_path):
                        if tier_file.endswith('.json'):
                            files_found += 1

                            # Construct file path
                            file_path = os.path.join(hmo_path, tier_file)

                            # Get metadata from path
                            service = service_dir
                            hmo = hmo_dir
                            tier = tier_file.replace('.json', '')

                            print(f"Processing file: {file_path}")

                            # Load JSON file
                            json_data = load_json_file(file_path)
                            if not json_data:
                                continue

                            # Prepare text for embedding
                            text = prepare_text_for_embedding(json_data)

                            # Create embedding
                            embedding = create_embedding(text)
                            if not embedding:
                                continue

                            files_embedded += 1

                            # Create a record
                            record = {
                                'id': f"{service}_{hmo}_{tier}",
                                'service': service,
                                'hmo': hmo,
                                'tier': tier,
                                'file_path': file_path,
                                'text': text,
                                'embedding': embedding,
                                'json_data': json_data
                            }

                            embedding_records.append(record)
                            print(f"Embedded: {file_path}")

    print(f"Found {files_found} files, successfully embedded {files_embedded} files")
    return embedding_records


def save_embeddings_to_csv(embedding_records, output_file):
    """Save embedding records to a CSV file."""
    # Create a DataFrame
    records_for_df = []
    for record in embedding_records:
        # Create a copy without the embedding and json_data
        df_record = {
            'id': record['id'],
            'service': record['service'],
            'hmo': record['hmo'],
            'tier': record['tier'],
            'file_path': record['file_path'],
            'text': record['text'],
        }
        records_for_df.append(df_record)

    df = pd.DataFrame(records_for_df)
    df.to_csv(output_file, index=False)
    print(f"Saved embedding metadata to {output_file}")

    return df


def save_embeddings_to_numpy(embedding_records, output_file):
    """Save embedding vectors to a NumPy file."""
    if not embedding_records:
        print("No embeddings to save.")
        # Create an empty array
        embeddings_array = np.array([])
    else:
        # Extract embeddings
        embeddings = [record['embedding'] for record in embedding_records]
        # Convert to NumPy array
        embeddings_array = np.array(embeddings)

    # Save to file
    np.save(output_file, embeddings_array)
    print(f"Saved embedding vectors to {output_file}")


def save_json_data(embedding_records, output_dir):
    """Save the original JSON data to files for quick access."""
    os.makedirs(output_dir, exist_ok=True)

    for record in embedding_records:
        output_file = os.path.join(output_dir, f"{record['id']}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(record['json_data'], f, ensure_ascii=False, indent=2)

    print(f"Saved JSON data files to {output_dir}")


if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed_data'
    OUTPUT_DIR = BASE_DIR / 'data' / 'embeddings'

    print(f"Base directory: {BASE_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Embed knowledge base
    print("Embedding knowledge base files...")
    embedding_records = embed_knowledge_base(PROCESSED_DATA_DIR)

    # Save results
    metadata_csv = os.path.join(OUTPUT_DIR, 'embeddings_metadata.csv')
    embeddings_npy = os.path.join(OUTPUT_DIR, 'embeddings.npy')
    json_dir = os.path.join(OUTPUT_DIR, 'json_data')

    save_embeddings_to_csv(embedding_records, metadata_csv)
    save_embeddings_to_numpy(embedding_records, embeddings_npy)
    save_json_data(embedding_records, json_dir)

    print(f"Embedded {len(embedding_records)} files from the knowledge base.")