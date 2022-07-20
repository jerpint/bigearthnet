import logging
import os
import pathlib
import random
import socket
import sys

import numpy as np
import torch
from git import InvalidGitRepositoryError, Repo
from omegaconf import DictConfig, OmegaConf
from pip._internal.operations import freeze

logger = logging.getLogger(__name__)


def set_seed(seed):  # pragma: no cover
    """Set the provided seed in python/numpy/DL framework.

    :param seed: (int) the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_info(script_location):  # pragma: no cover
    """Find the git hash for the running repository.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :return: (str) the git hash for the repository of the provided script.
    """
    if not script_location.endswith(".py"):
        raise ValueError("script_location should point to a python script")
    repo_folder = os.path.dirname(script_location)
    try:
        repo = Repo(repo_folder, search_parent_directories=True)
        commit_hash = repo.head.commit
        branch_name = repo.active_branch
    except (InvalidGitRepositoryError, ValueError):
        commit_hash = "git repository not found"
    return commit_hash, branch_name


def get_exp_details(cfg):  # pragma: no cover
    """Will log the experiment details to both screen logger and mlflow.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :param args: the argparser object.
    """
    # Log and save the config used for reproducibility
    script_path = pathlib.Path(str(sys.modules["__main__"].__file__)).resolve()
    git_hash, git_branch_name = get_git_info(str(script_path))
    hostname = socket.gethostname()
    dependencies = freeze.freeze()
    dependencies_str = "\n".join([d for d in dependencies])
    details = f"""
              config: {OmegaConf.to_yaml(cfg)}
              hostname: {hostname}
              git code hash: {git_hash}
              git branch name: {git_branch_name}
              dependencies: {dependencies_str}
              """
    return details
