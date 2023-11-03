import yaml

DATA_SET_KEY_NAME_MAP = {"plain": "Plain 1807 station data",
                         "analog_3p": "Enhanced input with 3% sampled from ARM",
                         "analog_3p_WT": "Enhanced input with 3% sampled from ARM, restricted by Weather Types",
                         }
DATA_SET_NAME_KEY_MAP = {v: k for k, v in DATA_SET_KEY_NAME_MAP.items()}


def get_config():
    with open("config.yaml", "r") as cnf:
        data = yaml.load(cnf, Loader=yaml.FullLoader)
        return data
