from datasets import load_dataset
import os
from astrapy import DataAPIClient

# Load API token from environment variable
ASTRA_DB_URI = os.getenv("ASTRA_DB_URI")

# Connect to Astra DB
client = DataAPIClient(ASTRA_DB_URI)
db = client.get_database_by_api_endpoint(
    "https://c690d8f6-f049-4761-967b-4914b961c961-us-east-2.apps.astra.datastax.com"
)

# Ensure the collection exists before inserting data
collection_name = "vector"  # Use lowercase as required by Astra DB
existing_collections = db.list_collection_names()

if collection_name not in existing_collections:
    db.create_collection(collection_name)

collection = db.get_collection(collection_name)

def load_oasst1_dataset():
    """Loads OpenAssistant dataset from Hugging Face"""
    dataset = load_dataset("OpenAssistant/oasst1")
    train_data = dataset["train"]
    print(f"Total Training Samples: {len(train_data)}")
    return train_data

def store_data_in_astra():
    """Stores sample training data into Astra DB"""
    train_data = load_oasst1_dataset()

    # Insert first 5 samples into Astra DB
    sample_data = train_data.select(range(5))
    for sample in sample_data:
        collection.insert_one(dict(sample))

    print("Sample data uploaded to Astra DB!")

if __name__ == "__main__":
    store_data_in_astra()
