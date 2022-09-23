import logging
import os

from git import Repo
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    logger.info(f"Setting seed to: {seed}")
    pl.utilities.seed.seed_everything(seed=seed, workers=False)


def get_git_info(script_location):
    """Find the git hash for the running repository.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :return: (str) the git hash for the repository of the provided script.
    """
    repo_folder = os.path.dirname(script_location)
    repo = Repo(repo_folder, search_parent_directories=True)
    commit_hash = repo.head.commit

    try:
        branch_name = repo.active_branch
    except TypeError:
        # Happens on e.g. github actions
        branch_name = "detached head"
    return commit_hash, branch_name
