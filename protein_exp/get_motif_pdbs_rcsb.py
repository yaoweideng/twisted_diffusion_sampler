import os
import pandas as pd
import requests
from pathlib import Path

def download_pdb(pdb_id: str, output_dir: str) -> bool:
    """
    Download a PDB file from RCSB.
    
    Args:
        pdb_id: The PDB ID to download
        output_dir: Directory to save the PDB file
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"Successfully downloaded {pdb_id}")
        return True
    except Exception as e:
        print(f"Failed to download {pdb_id}: {str(e)}")
        return False

def main():
    # Create output directory
    output_dir = "mb_rcsb_pdb"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read test cases CSV
    csv_path = "test_cases.csv"
    df = pd.read_csv(csv_path)
    
    # Get unique PDB IDs
    unique_pdbs = df['pdb_id'].unique()
    
    # Download each PDB
    successful_downloads = 0
    for pdb_id in unique_pdbs:
        if download_pdb(pdb_id, output_dir):
            successful_downloads += 1
    
    print(f"\nDownload summary:")
    print(f"Total PDBs attempted: {len(unique_pdbs)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {len(unique_pdbs) - successful_downloads}")

if __name__ == "__main__":
    main()
