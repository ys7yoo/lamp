# What is this?

This project contains scripts to reproduce experiments from the paper
[AMP-Inspired Deep Networks for Sparse Linear Inverse Problems](http://ieeexplore.ieee.org/document/7934066/)
by 
[Mark Borgerding](mailto://borgerding.7@osu.edu)
,
[Phil](mailto://schniter.1@osu.edu)
[Schniter](http://www2.ece.ohio-state.edu/~schniter)
, and [Sundeep Rangan](http://engineering.nyu.edu/people/sundeep-rangan).
To appear in IEEE Transactions on Signal Processing.
See also the related [preprint](https://arxiv.org/pdf/1612.01183)

# The Problem of Interest

Briefly, the _Sparse Linear Inverse Problem_ is the estimation of an unknown signal from indirect, noisy, underdetermined measurements by exploiting the knowledge that the signal has many zeros.  We compare various iterative algorithmic approaches to this problem and explore how they benefit from loop-unrolling and deep learning.

# Overview

The included scripts 
- are generally written in python and require [TensorFlow](http://www.tensorflow.org),
- work best with a GPU,
- generate synthetic data as needed,
- are known to work with CentOS 7 Linux and TensorfFlow 1.1,
- are sometimes be written in octave/matlab .m files.

##  If you are just looking for an implementation of VAMP ...

You might prefer the Matlab code in [GAMP](https://sourceforge.net/projects/gampmatlab/)/code/VAMP/ 
or the python code in [Vampyre](https://github.com/GAMPTeam/vampyre).

# Description of Files

## [save_problem.py](save_problem.py) 

Creates numpy archives (.npz) and matlab (.mat) files with (y,x,A) for the sparse linear problem y=Ax+w.
These files are not really necessary for any of the deep-learning scripts, which generate the problem on demand.
They are merely provided for better understanding the specific realizations used in the experiments.

```
$ python3 save_problem.py
saving problem_Giid.mat,problem_Giid.npz norm(x)=224.5516663 norm(y)=224.0477230
saving problem_k15.mat,problem_k15.npz norm(x)=224.4524994 norm(y)=224.3352588
saving problem_k100.mat,problem_k100.npz norm(x)=224.2500458 norm(y)=224.0362651
saving problem_rap1.mat,problem_rap1.npz norm(x)=276.4882507 norm(y)=289.9351196
saving problem_rap2.mat,problem_rap2.npz norm(x)=89.6196136 norm(y)=90.3195496
```

## [ista_fista_amp.m](ista_fista_amp.m)

Using the .mat files created by save_problem.py, this octave/matlab script tests the performance of non-learned algorithms ISTA, FISTA, and AMP.

e.g.
```
>> ista_fista_amp
loaded Gaussian A problem
AMP reached NMSE=-35dB at iteration 26
AMP terminal NMSE=-36.7865 dB
FISTA reached NMSE=-35dB at iteration 202
FISTA terminal NMSE=-36.8365 dB
ISTA reached NMSE=-35dB at iteration 3754
ISTA terminal NMSE=-36.8367 dB
```

![MMSE-vs-itr.png](results/MMSE-vs-itr.png)

![QQ.png](results/QQ.png)


## [LISTA.py](LISTA.py)

This is an example implementation of LISTA _Learned Iterative Soft Thresholding Algorithm_ by (Gregor&LeCun, 2010 ICML).

## [LAMP.py](LAMP.py)

Example of Learned AMP (LAMP) with a variety of shrinkage functions.

## [LVAMP.py](LVAMP.py)

Example of Learned Vector AMP (LVAMP).

