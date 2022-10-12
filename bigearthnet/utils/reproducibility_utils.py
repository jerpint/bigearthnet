import logging
import os

import pytorch_lightning as pl
from git import Repo

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set a seed for reproduciblity."""
    logger.info(f"Setting seed to: {seed}")
    pl.utilities.seed.seed_everything(seed=seed, workers=False)


def get_git_info(script_location):
    """Find the git hash for the running repository."""
    repo_folder = os.path.dirname(script_location)
    repo = Repo(repo_folder, search_parent_directories=True)
    commit_hash = repo.head.commit

    try:
        branch_name = repo.active_branch
    except TypeError:
        # Happens on e.g. github actions
        branch_name = "detached head"
    return commit_hash, branch_name
