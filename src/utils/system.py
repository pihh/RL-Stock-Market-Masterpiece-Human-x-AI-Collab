import os
import torch
import random
import requests
import numpy as np
import multiprocessing

from os.path import join, dirname
from dotenv import load_dotenv

SEED = 314
NTFY_TOPIC ="pihh-trading-test"

def notify(message,title="Something happened",topic=NTFY_TOPIC,level="info"):
    
    LEVELS = dict()
    LEVELS['info'] = "loudspeaker"
    LEVELS['success'] = "+1,tada"
    LEVELS['warning']= "warning,skull"
    LEVELS['danger'] = "-1,rotating_light"
    
    requests.post(f"https://ntfy.sh/{topic}",
        data=message,
        headers={
            "Title": title,
            #"Priority": "urgent",
            "Tags": LEVELS[level]
        })


def load_environment():
    """
    Load .env variables
    """
    dotenv_path = join(dirname(__file__),'../','../', '.env')
    load_dotenv(dotenv_path)

load_environment()


def get_system_variables():
    API_KEY = os.getenv("ALPACA_PAPER_API_KEY")
    API_SECRET = os.getenv("ALPACA_PAPER_API_SECRET")

    SAVE_DIR = "./data"
    OHLCV_DIR = f"{SAVE_DIR}/ohlcv/"
    TOKENS_DIR = f"{SAVE_DIR}/tokens/"
    LABELED_DIR = f"{SAVE_DIR}/labeled/"
    MODELS_DIR = f"{SAVE_DIR}/models/"
    CHECKPOINT_DIR = f"{SAVE_DIR}/checkpoint/"
    LOG_DIR = f"{SAVE_DIR}/logs/"
    TENSORBOARD_DIR = f"{LOG_DIR}/tensorboard/"

    for _dir in [SAVE_DIR, OHLCV_DIR, TOKENS_DIR, LABELED_DIR, MODELS_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR]:
        os.makedirs(_dir, exist_ok=True)

    return {
        "ALPACA_PAPER_API_KEY": API_KEY,
        "ALPACA_PAPER_API_SECRET": API_SECRET,
        "SAVE_DIR": SAVE_DIR,
        "OHLCV_DIR": OHLCV_DIR,
        "TOKENS_DIR": TOKENS_DIR,
        "LABELED_DIR": LABELED_DIR,
        "MODELS_DIR": MODELS_DIR,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "TENSORBOARD_DIR": TENSORBOARD_DIR,
        "SEED": SEED
    }


def set_seed(seed=314,environment="development"):
    
    if environment != "development":
        seed = random.randint(0, 2**16 - 1) # Fugi do 32 não sei pq não me parece bem
    
    global SEED 
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return SEED
    


def boot(seed=314):
    """Sets up the random seed across the system.
    Keep a fixed seed to proof of concepts and notebooks with the studies
    Change the seed on every iteration of pre live releases so it's possible to 
    check if the project is realy performing well on something closer to
    a real environemnt

    Args:
        seed (int, optional): _description_. Defaults to 314.

    Returns:
        _type_: _description_
    """
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    set_seed(seed=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def get_seed(environment="development"):
    global SEED    
    if environment !="development":
        return set_seed(seed=SEED,environment=environment) 
    else:
        SEED

def log(title, sentences):

    title = f"# {title} "
    strlen = len(title)
    charlen = 100 - strlen

    if (charlen > strlen):
        title = "="*charlen
    print("")
    print(f"# {title}")
    if isinstance(sentences, list):
        for sentence in sentences:
            print(f"# {sentence}")

    else:
        print(f"# {sentences}")


def parse_n_environments(n=8):

    # Number of CPU cores
    max_envs = multiprocessing.cpu_count()

    # Ensure it will be between 1 and max_envs
    n_envs = max(1, min(n, max_envs))

    print(f"Using {n_envs} environments out of {max_envs} CPU cores available.")

    return n_envs
