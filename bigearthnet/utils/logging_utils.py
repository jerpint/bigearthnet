import logging
import mlflow
import os
import socket

from pip._internal.operations import freeze
from git import InvalidGitRepositoryError, Repo
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE

logger = logging.getLogger(__name__)


class LoggerWriter:  # pragma: no cover
    """LoggerWriter.

    see: https://stackoverflow.com/questions/19425736/
    how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, printer):
        """__init__.

        Args:
            printer: (fn) function used to print message (e.g., logger.info).
        """
        self.printer = printer

    def write(self, message):
        """write.

        Args:
            message: (str) message to print.
        """
        if message != '\n':
            self.printer(message)

    def flush(self):
        """flush."""
        pass


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


def log_exp_details(script_location, args):  # pragma: no cover
    """Will log the experiment details to both screen logger and mlflow.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :param args: the argparser object.
    """
    git_hash = get_git_hash(script_location)
    hostname = socket.gethostname()
    dependencies = freeze.freeze()
    details = "\nhostname: {}\ngit code hash: {}\ndata folder: {}\ndata folder (abs): {}\n\n" \
              "dependencies:\n{}".format(
                  hostname, git_hash, args.data, os.path.abspath(args.data),
                  '\n'.join(dependencies))
    logger.info('Experiment info:' + details + '\n')
    mlflow.set_tag(key=MLFLOW_RUN_NOTE, value=details)
