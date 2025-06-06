import os
import pandas as pd
import requests
from pathlib import Path

def download_pdb(pdb_id: str, output_path: str) -> bool:
    """
    Download a PDB file from RCSB.
    
    Args:
        pdb_id: The PDB ID to download
        output_path: Full path where to save the PDB file
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"Successfully downloaded {pdb_id} to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to download {pdb_id}: {str(e)}")
        return False

def main():
    # Read both CSV files
    test_cases_df = pd.read_csv("test_cases.csv")
    mb_test_cases_df = pd.read_csv("motif_scaffolding/mb_test_cases.csv")
    
    # Create output directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    mb_rcsb_pdb_dir = os.path.join(current_dir, "mb_rcsb_pdb")
    os.makedirs(mb_rcsb_pdb_dir, exist_ok=True)
    
    # Download each PDB with target name
    successful_downloads = 0
    for idx, row in test_cases_df.iterrows():
        pdb_id = row['pdb_id']
        target = mb_test_cases_df.iloc[idx]['target']
        output_path = os.path.join(mb_rcsb_pdb_dir, f"{target}.pdb")
        
        if download_pdb(pdb_id, output_path):
            successful_downloads += 1
    
    # Create new dataframe with specified columns
    new_df = pd.DataFrame({
        'pdb_id': test_cases_df['pdb_id'],
        'contig': test_cases_df['motif_residues'].str.replace(';', ','),
        'idcs_to_redesign': test_cases_df['redesign_idcs'],
        'length_fixed': mb_test_cases_df['length_fixed'],
        'length': mb_test_cases_df['length'],
        'target': mb_test_cases_df['target'],
        'motif_path': [os.path.join(mb_rcsb_pdb_dir, f"{target}.pdb") for target in mb_test_cases_df['target']]
    })
    
    # Save to new CSV file
    new_df.to_csv("rcsb_test_cases.csv", index=False)
    print(f"\nDownload summary:")
    print(f"Total PDBs attempted: {len(test_cases_df)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {len(test_cases_df) - successful_downloads}")
    print("\nCreated rcsb_test_cases.csv successfully")

if __name__ == "__main__":
    main()
