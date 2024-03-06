from joblib import dump, load
import time
import os

def save_scaler(scaler_to_save = None,
               scaler_type = None,
               path_to_save = "/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/api/saved_scalers"
              ):
    """
    Persist trained model locally on the hard drive at f"{path_to_save/scaler_type/f"{timestamp}.joblib"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save scaler locally
    scaler_path_file = os.path.join(f"{path_to_save}/{scaler_type}_{timestamp}.joblib")

    dump(scaler_to_save, scaler_path_file)

    print(f"✅ Scaler saved locally at {scaler_path_file}")

# Load scaler
def load_scaler(path = '/home/jana/code/Klara-haas/brain_proteomics_project/brain_proteomics/api/saved_scalers',
               file = 'MinMax_20240306-102844.joblib'
              ):
    path_file = f"{path}/{file}"

    scaler = load(path_file)
    return scaler
