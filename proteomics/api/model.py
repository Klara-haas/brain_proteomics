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



def load_model(path = '~/api/saved_models',
               file = 'SGDclassifier_20240305-135742.joblib'
              ):
    path_file = f"{path}/{file}"

    model = load(path_file)
    return model
