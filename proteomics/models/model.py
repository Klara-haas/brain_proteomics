from joblib import dump, load
import time
import os


def save_model(model_to_save = None,
               model_type = None,
               path_to_save = "~/api/saved_models"
              ):
    """
    Persist trained model locally on the hard drive at f"{path_to_save/model_type_{timestamp}.joblib"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path_file = os.path.join(f"{path_to_save}/{model_type}_{timestamp}.joblib")

    dump(model_to_save, model_path_file)

    print(f"✅ Model saved locally at {model_path_file}")
