import yaml

DATA_SET_KEY_NAME_MAP = {"plain": "1807",
                         "arm": "1807, ARM-Enhanced",
                         "arm_wt": "1807, ARM restricted by Weather Types",
                         }
DATA_SET_NAME_KEY_MAP = {v: k for k, v in DATA_SET_KEY_NAME_MAP.items()}
DATA_SET_DESCRIPTIONS = {"plain": "Reconstruction based solely on the station data from 1807",
                         "arm": "Input for the reconstruction model was enhanced with 3% randomly sampled cells "
                                "using the Analog Resampling Method",
                         "arm_wt": "Additionally restricting the analogs on weather types (p=90%)",
                         }

CITIES = {
    'ARM': 'Armagh',
    'BAR': 'Barcelona',
    'BRL': 'Berlin',
    'CAD': 'Cadiz',
    'CBT': 'Central Belgium',
    'CET': 'Central England',
    'GVE': 'Geneva',
    'HAA': 'Haarlem',
    'HOH': 'Hohenpeissenberg',
    'KAR': 'Karlsruhe',
    'LON': 'London',
    'MIL': 'Milano',
    'MUL': 'Mulhouse',
    'PAD': 'Padova',
    'PAR': 'Paris',
    'PRA': 'Prag',
    'ROV': 'Rovereto',
    'SHA': 'Schaffhausen',
    'STK': 'Stockholm',
    'STP': 'St. Petersburg',
    'TOR': 'Torino',
    'UPP': 'Uppsala',
    'VAL': 'Valencia',
    'VIL': 'Vilnius',
    'WAR': 'Warschau',
    'WRO': 'Wroclaw',
    'YLI': 'Yilitornio',
    'ZIT': 'Zitenice'
}

# Define reduced size window
BASE_LAT_START = 67
BASE_LAT_END = 36  # Considered exclusive
BASE_LON_START = -22
BASE_LON_END = 41  # Considered exclusive


def get_config():
    with open("config.yaml", "r") as cnf:
        data = yaml.load(cnf, Loader=yaml.FullLoader)
        return data
