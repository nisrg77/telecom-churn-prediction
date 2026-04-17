import pandas as pd
import os

# --- STEP 1: FETCH REAL DATA ---
# Instead of generating "fake" synthetic data, we download a professional dataset 
# used by data scientists worldwide for telecom churn analysis.
def fetch_data():
    print("Fetching real-world IBM Telco Customer Churn dataset...")
    
    # Official URL for the dataset hosted on IBM's GitHub
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    try:
        # pd.read_csv downloads the file directly into a memory buffer (DataFrame)
        df = pd.read_csv(url)
        
        # We ensure a 'data' folder exists to keep our workspace tidy
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Saving the data locally so we don't have to download it every time we train
        save_path = 'data/telco_churn_real.csv'
        df.to_csv(save_path, index=False)
        
        print(f"Dataset successfully saved to {save_path}")
        print(f"Total Profiles: {len(df)}")
        print(f"Features Retrieved: {', '.join(df.columns)}")
        
    except Exception as e:
        print(f"Error fetching dataset: {e}")

# entry point: starts the script when executed directly
if __name__ == '__main__':
    fetch_data()
