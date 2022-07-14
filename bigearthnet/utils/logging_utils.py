import logging
import os
import socket

from pip._internal.operations import freeze
from git import InvalidGitRepositoryError, Repo

logger = logging.getLogger(__name__)


def get_git_hash(script_location):  # pragma: no cover
    """Find the git hash for the running repository.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :return: (str) the git hash for the repository of the provided script.
    """
    if not script_location.endswith('.py'):
        raise ValueError('script_location should point to a python script')
    repo_folder = os.path.dirname(script_location)
    try:
        repo = Repo(repo_folder, search_parent_directories=True)
        commit_hash = repo.head.commit
    except (InvalidGitRepositoryError, ValueError):
        commit_hash = 'git repository not found'
    return commit_hash


def log_exp_details(script_location):  # pragma: no cover
    """Will log the experiment details to both screen logger and mlflow.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :param args: the argparser object.
    """
    git_hash = get_git_hash(script_location)
    hostname = socket.gethostname()
    dependencies = freeze.freeze()
    details = f"""
              hostname: {hostname}
              git code hash: {git_hash}
              dependencies: {dependencies}
              """
    logger.info('Experiment info:' + details + '\n')
