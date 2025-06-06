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
    # Read both CSV files
    test_cases_df = pd.read_csv("test_cases.csv")
    mb_test_cases_df = pd.read_csv("motif_scaffolding/mb_test_cases.csv")
    
    # Create new dataframe with specified columns
    new_df = pd.DataFrame({
        'pdb_id': test_cases_df['pdb_id'],
        'motif_residues': test_cases_df['motif_residues'].str.replace(';', ','),
        'idcs_to_redesign': test_cases_df['redesign_idcs'],
        'length_fixed': mb_test_cases_df['length_fixed'],
        'length': mb_test_cases_df['length'],
        'target': mb_test_cases_df['target'],
        'motif_path': mb_test_cases_df['motif_path']
    })
    
    # Save to new CSV file
    new_df.to_csv("rcsb_test_cases.csv", index=False)
    print("Created rcsb_test_cases.csv successfully")

if __name__ == "__main__":
    main()
