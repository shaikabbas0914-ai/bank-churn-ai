import os
import urllib.request

def download_dataset():
    data_dir = "data"
    dataset_url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"
    destination_path = os.path.join(data_dir, "Churn_Modelling.csv")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        print(f"Creating directory: {data_dir}...")
        os.makedirs(data_dir)
        
    print(f"Downloading dataset from: {dataset_url}...")
    try:
        # Use urllib to download the CSV file
        urllib.request.urlretrieve(dataset_url, destination_path)
        print(f"Dataset successfully downloaded and saved to: {destination_path}")
    except Exception as e:
        print(f"An error occurred during dataset download: {e}")
        print("Please check your internet connection or download the file manually.")

if __name__ == "__main__":
    download_dataset()
