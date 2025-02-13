from pymongo import MongoClient

# Replace with your Astra DB connection string
ASTRA_DB_URI = "your_connection_string_here"

# Connect to the database
client = MongoClient(ASTRA_DB_URI)
db = client["Finetune"]  # Database name
collection = db["Vector"]  # Collection name

def insert_sample_data():
    """Insert sample data into Astra DB"""
    sample_data = {"model_name": "Llama2", "parameters": "7B", "fine_tuned": False}
    collection.insert_one(sample_data)
    print("Sample data inserted successfully!")

if __name__ == "__main__":
    insert_sample_data()
