# BigEarthNet


BigEarthNet classification


* Free software: MIT license



## Instructions to setup the project

### Install the dependencies:
(remember to activate the virtual env if you want to use one)
Add new dependencies (if needed) to setup.py.

    pip install -e .

### Add git:

    git init

### Setup pre-commit hooks:
These hooks will:
* validate flake8 before any commit
* check that jupyter notebook outputs have been stripped

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -

### Commit the code

    git add .
    git commit -m 'first commit'

### Link github to your local repository
Go on github and follow the instructions to create a new project.
When done, do not add any file, and follow the instructions to
link your local git to the remote project, which should look like this:
(PS: these instructions are reported here for your convenience.
We suggest to also look at the GitHub project page for more up-to-date info)

    git remote add origin git@github.com:jerpint-mila/bigearthnet.git
    git branch -M main
    git push -u origin main

### Setup Continuous Integration

Continuous integration will run the following:
- Unit tests under `tests`.
- End-to-end test under `exmaples/local`.
- `flake8` to check the code syntax.
- Checks on documentation presence and format (using `sphinx`).

We support the following Continuous Integration providers.
Check the following instructions for more details.

#### GitHub Actions

Github actions are already configured in `.github/workflows/tests.yml`.
Github actions are already enabled by default when using Github, so, when
pushing to github, they will be executed automatically for pull requests to
`main` and to `develop`.

#### Travis

Travis is already configured in (`.travis.yml`).

To enable it server-side, just go to https://travis-ci.com/account/repositories and click
` Manage repositories on GitHub`. Give the permission to run on the git repository you just created.

Note, the link for public project may be https://travis-ci.org/account/repositories .

#### Azure

Azure Continuous Integration is already configured in (`.azure_pipeline.yml`).

To enable it server-side, just in azure and select `.azure_pipeline.yml` as the 
configuration one for Continuous Integration.

## Running the code

### Run the tests
Just run (from the root folder):

    pytest

### Run the code/examples.
Note that the code should already compile at this point.

Running examples can be found under the `examples` folder.

In particular, you will find examples for:
* local machine (e.g., your laptop).
* a slurm cluster.

For both these cases, there is the possibility to run with or without Orion.
(Orion is a hyper-parameter search tool - see https://github.com/Epistimio/orion -
that is already configured in this project)

#### Run locally

For example, to run on your local machine without Orion:

    cd examples/local
    sh run.sh

This will run a simple MLP on a simple toy task: sum 5 float numbers.
You should see an almost perfect loss of 0 after a few epochs.

Note you have two new folders now:
* output: contains the models and a summary of the results.
* mlruns: produced by mlflow, contains all the data for visualization.
You can run mlflow from this folder (`examples/local`) by running
`mlflow ui`.

#### Run on a remote cluster (with Slurm)

First, bring you project on the cluster (assuming you didn't create your
project directly there). To do so, simply login on the cluster and git
clone your project:

    git clone git@github.com:jerpint-mila/bigearthnet.git

Then activate your virtual env, and install the dependencies:

    cd bigearthnet
    pip install -e .

To run with Slurm, just:

    cd examples/slurm
    sh run.sh

Check the log to see that you got an almost perfect loss (i.e., 0).

#### Measure GPU time (and others) on the Mila cluster

You can track down the GPU time (and other resources) of your jobs by
associating a tag to the job (when using `sbatch`).
To associate a tag to a job, replace `my_tag` with a proper tag,
and uncomment the line (i.e., remove one #) from the line:

    ##SBATCH --wckey=my_tag

This line is inside the file `examples/slurm_mila/to_submit.sh`.

To get a sumary for a particular tag, just run:

    sacct --allusers --wckeys=my_tag --format=JobID,JobName,Start,Elapsed -X -P --delimiter=','

(again, remember to change `my_tag` into the real tag name)

#### GPU profiling on the Mila cluster

It can be useful to monitor and profile how you utilise your GPU (usage, memory, etc.). For the time being, you can only monitor your profiling in real-time from the Mila cluster, i.e. while your experiments are running. To monitor your GPU, you need to setup port-forwarding on the host your experiments are running on. This can be done in the following way:

Once you have launched your job on the mila cluster, open the log for your current experiment:

`head logs/bigearthnet__<your_slurm_job_id>.err`

You should see printed in the first few lines the hostname of your machine, e.g.,

```
INFO:bigearthnet.utils.logging_utils:Experiment info:
hostname: leto35
git code hash: a51bfc5447d188bd6d31fac3afbd5757650ef524
data folder: ../data
data folder (abs): /network/tmp1/bronzimi/20191105_cookiecutter/bigearthnet/examples/data
```

In a separate shell on your local computer, run the following command:

`ssh -L 19999:<hostname>.server.mila.quebec:19999 <username>@login.server.mila.quebec -p 2222` 

where `<username>` is your user name on the Mila cluster and `<hostname>` is the name of the machine your job is currenty running on (`leto35` in our example). You can then navigate your local browser to `http://localhost:19999/` to view the ressources being used on the cluster and monitor your job. You should see something like this:

![image](https://user-images.githubusercontent.com/18450628/88088807-fe2acd80-cb58-11ea-8ab2-bd090e8a826c.png)

#### Run with Orion on the Slurm cluster

This example will run orion for 2 trials (see the orion config file).
To do so, go into `examples/slurm_orion`.
Here you can find the orion config file (`orion_config.yaml`), as well as the config
file (`config.yaml`) for your project (that contains the hyper-parameters).

In general, you will want to run Orion in parallel over N slurm jobs.
To do so, simply run `sh run.sh` N times.

When Orion has completed the trials, you will find the orion db file and the
mlruns folder (i.e., the folder containing the mlflow results).

You will also find the output of your experiments in `orion_working_dir`, which
will contain a folder for every trial.
Inside these folders, you can find the models (the best one and the last one), the config file with
the hyper-parameters for this trial, and the log file.

You can check orion status with the following commands:
(to be run from `examples/slurm_orion`)

    export ORION_DB_ADDRESS='orion_db.pkl'
    export ORION_DB_TYPE='pickleddb'
    orion status
    orion info --name my_exp

### Building docs:

To automatically generate docs for your project, cd to the `docs` folder then run:

    make html

To view the docs locally, open `docs/_build/html/index.html` in your browser.


## YOUR PROJECT README:

* __TODO__
