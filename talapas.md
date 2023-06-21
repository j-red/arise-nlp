# Talapas Overview

This markdown file contains the information I found useful when working with the University of Oregon Talapas High Performance Computing cluster. It is not necessary for the final project submission, but may be useful if attempting to use Talapas in the future.

# Working with Talapas

https://hpcrcf.atlassian.net/wiki/spaces/TCP/pages/7312376/Quick+Start+Guide

I recommend setting an environment variable `PIRG` instead of typing it in each time.

# Running an interactive job

https://hpcrcf.atlassian.net/wiki/spaces/TCP/pages/7089568/How-to+Start+an+Interactive+Job

`srun --account=$PIRG --pty --time=240 bash`

If you do not specify a time limit (in minutes), it will assume 24 hours and wait until it can allocate that much time for you. 
Other parameters include `--partition=short`, `--mem=1024M`, and more.

# Job Parameters

To see a list of parameters, run `scontrol show config` to see the default settings.

`--time <val>` or `-t <val>`: 
Time limit for the duration of rental of the compute node. A time limit of zero requests that no time limit be imposed. Acceptable time formats for the `-t` time parameter include `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes` and `days-hours:minutes:seconds`.

`--mem <val>`: 
The amount of memory allocated to each node. Default units are megabytes. Different units can be specified using the suffix [K|M|G|T].

# Estimating requirement usage

In order to help estimate your job's requirements, you can use sacct to see what prior jobs required

`sacct -j JOBID --format=JobID,JobName,ReqMem,MaxRSS,Elapsed`
`sacct -j $SLURM_JOB_ID --format=JobID,JobName,ReqMem,MaxRSS,Elapsed`

where JOBID is the numeric ID of a job that previously completed.  This will produce output like this:

```
sacct -j 301111 --format=JobID,JobName,ReqMem,MaxRSS,Elapsed
       JobID    JobName     ReqMem     MaxRSS    Elapsed
------------ ---------- ---------- ---------- ----------
301111        myjobname     3800Mc              16:00:28
301111.batch      batch     3800Mc    197180K   16:00:30
301111.exte+     extern     3800Mc      2172K   16:00:29
```

You can also use `seff <JOBID>` to see the current usage of a job. 
Note that `sacct` will NOT work from a compute node; only the login nodes.
Run `sacct` to see a list of running jobs.

# Debug Info

When running an interactive job, SLURM sets some environment variables according to what job is running.

To see how much time is remaining for the current job:
`squeue --job $SLURM_JOB_ID -o %L | tail -1f`

To connect to an **owned** compute node, run `hostname` from the node, and then `ssh` to that host (e.g., `ssh n027`) from the login node.
You may also wish to port forward (e.g., `ssh -L 5000:localhost:5000 <host>`) to connect to ports hosted on the compute node (e.g., the ARISE webserver or backend interface).
