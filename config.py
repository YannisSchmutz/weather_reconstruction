import yaml

DATA_SET_KEY_NAME_MAP = {"plain": "1807",
                         "analog_3p": "1807, Analog 3%",
                         "analog_3p_WT": "1807, Analog 3%, restricted by Weather Types",
                         }
DATA_SET_NAME_KEY_MAP = {v: k for k, v in DATA_SET_KEY_NAME_MAP.items()}
DATA_SET_DESCRIPTIONS = {"plain": "Reconstruction based solely on the station data from 1807",
                         "analog_3p": "Input for the reconstruction model was enhanced with 3% randomly sampled cells "
                                      "using the Analog Resampling Method",
                         "analog_3p_WT": "Additionally restricting the analogs on weather types (p=90%)",
                         }


def get_config():
    with open("config.yaml", "r") as cnf:
        data = yaml.load(cnf, Loader=yaml.FullLoader)
        return data
