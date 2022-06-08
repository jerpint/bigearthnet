import random


def generate_for(prefix, size):
    """generate_for.

    Args:
        prefix (str): The data split to target, i.e. "train" or "dev.
        size (int): Number of examples to generate.
    """
    with open('{}.input'.format(prefix), 'w') as inp_stream:
        with open('{}.target'.format(prefix), 'w') as tar_stream:
            for _ in range(size):
                inp = [random.uniform(0, 1) for _ in range(5)]
                tar = sum(inp)
                inp_stream.write(' '.join([str(x) for x in inp]) + '\n')
                tar_stream.write(str(tar) + '\n')


generate_for('train', 1000)
generate_for('dev', 100)
