##What's it?

In this folder very simple cuda scripts are acumulated and pretty simplest workflow is organized:
* An universal makefile is written - you pass the target name by make targ=<target_name>, assuming your source file has name <target_name>.cu
* Running script takes <target_name>, compiles target <target_name> and runs it
* Results of run are collected in a current folder with a name like slurm-<job_number>.out

##How to run?

```bash
sbatch ../run_gpu.sh test
```

##How to look throught the result?

```
cat slurm-<job_number>.out
```

after job is finished or
```
less slurm-<job_number>.out
```

and then `Cntl+F`.
