import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def rsync_folder(source, target):  # pragma: no cover
    """Uses rsync to copy the content of source into target.

    :param source: (str) path to the source folder.
    :param target: (str) path to the target folder.
    """
    if not os.path.exists(target):
        os.makedirs(target)

    logger.info('rsyincing {} to {}'.format(source, target))
    subprocess.check_call(["rsync", "-avzq", source, target])
