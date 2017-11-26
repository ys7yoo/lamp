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

```
$ python LISTA.py 
norms xval:224.5516510 yval:224.0477230
Linear trainrate=0.5 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-1.295706 dB (best=-1.295706)
i=1000   nmse=-2.978656 dB (best=-2.978656)
i=2000   nmse=-2.979118 dB (best=-2.980402)
i=3000   nmse=-2.977585 dB (best=-2.981049)
i=4000   nmse=-2.977192 dB (best=-2.981049)
i=5000   nmse=-2.978181 dB (best=-2.981049)
i=6000   nmse=-2.976449 dB (best=-2.981049)
i=7000   nmse=-2.977701 dB (best=-2.981049)
i=8000   nmse=-2.975074 dB (best=-2.981049)
Linear trainrate=0.1 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-2.975074 dB (best=-2.975074)
i=1000   nmse=-2.990523 dB (best=-2.991508)
i=2000   nmse=-2.990486 dB (best=-2.992042)
i=3000   nmse=-2.989711 dB (best=-2.992042)
i=4000   nmse=-2.991351 dB (best=-2.992042)
i=5000   nmse=-2.990159 dB (best=-2.992403)
i=6000   nmse=-2.991431 dB (best=-2.992403)
i=7000   nmse=-2.991723 dB (best=-2.992403)
i=8000   nmse=-2.989994 dB (best=-2.992403)
i=9000   nmse=-2.990943 dB (best=-2.992403)
i=10000  nmse=-2.991525 dB (best=-2.992403)
Linear trainrate=0.01 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-2.991525 dB (best=-2.991525)
i=1000   nmse=-2.993356 dB (best=-2.993408)
i=2000   nmse=-2.993673 dB (best=-2.993744)
i=3000   nmse=-2.993981 dB (best=-2.993981)
i=4000   nmse=-2.993938 dB (best=-2.994000)
i=5000   nmse=-2.993856 dB (best=-2.994000)
i=6000   nmse=-2.994006 dB (best=-2.994185)
i=7000   nmse=-2.994032 dB (best=-2.994185)
i=8000   nmse=-2.994238 dB (best=-2.994418)
i=9000   nmse=-2.994047 dB (best=-2.994418)
i=10000  nmse=-2.994208 dB (best=-2.994418)
i=11000  nmse=-2.994128 dB (best=-2.994418)
i=12000  nmse=-2.994221 dB (best=-2.994418)
i=13000  nmse=-2.994485 dB (best=-2.994515)
i=14000  nmse=-2.994253 dB (best=-2.994524)
i=15000  nmse=-2.994328 dB (best=-2.994679)
i=16000  nmse=-2.994405 dB (best=-2.994679)
i=17000  nmse=-2.994601 dB (best=-2.994679)
i=18000  nmse=-2.994339 dB (best=-2.994679)
i=19000  nmse=-2.994339 dB (best=-2.994679)
i=20000  nmse=-2.994187 dB (best=-2.994679)
LISTA T=1 extending lam_0:0
i=0      nmse=-3.678087 dB (best=-3.678087)
i=1000   nmse=-3.679837 dB (best=-3.679851)
i=2000   nmse=-3.679848 dB (best=-3.679851)
i=3000   nmse=-3.679848 dB (best=-3.679851)
i=4000   nmse=-3.679823 dB (best=-3.679851)
i=5000   nmse=-3.679851 dB (best=-3.679851)
i=6000   nmse=-3.679821 dB (best=-3.679851)
LISTA T=1 trainrate=0.5 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-3.679821 dB (best=-3.679821)
i=1000   nmse=-6.143229 dB (best=-6.143229)
i=2000   nmse=-6.360985 dB (best=-6.360985)
i=3000   nmse=-6.386578 dB (best=-6.386578)
i=4000   nmse=-6.388455 dB (best=-6.391043)
i=5000   nmse=-6.387405 dB (best=-6.392977)
i=6000   nmse=-6.391533 dB (best=-6.393390)
i=7000   nmse=-6.388214 dB (best=-6.396033)
i=8000   nmse=-6.387995 dB (best=-6.396033)
i=9000   nmse=-6.388549 dB (best=-6.396033)
i=10000  nmse=-6.390011 dB (best=-6.396033)
i=11000  nmse=-6.392407 dB (best=-6.396033)
i=12000  nmse=-6.388993 dB (best=-6.396033)
LISTA T=1 trainrate=0.1 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-6.388993 dB (best=-6.388993)
i=1000   nmse=-6.405227 dB (best=-6.405264)
i=2000   nmse=-6.405815 dB (best=-6.406590)
i=3000   nmse=-6.406335 dB (best=-6.408056)
i=4000   nmse=-6.406373 dB (best=-6.408601)
i=5000   nmse=-6.407241 dB (best=-6.408601)
i=6000   nmse=-6.406947 dB (best=-6.408601)
i=7000   nmse=-6.405915 dB (best=-6.408601)
i=8000   nmse=-6.407546 dB (best=-6.408601)
i=9000   nmse=-6.405352 dB (best=-6.408601)
LISTA T=1 trainrate=0.01 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-6.405352 dB (best=-6.405352)
i=1000   nmse=-6.406999 dB (best=-6.406999)
i=2000   nmse=-6.408278 dB (best=-6.408278)
i=3000   nmse=-6.409042 dB (best=-6.409042)
i=4000   nmse=-6.409610 dB (best=-6.409665)
i=5000   nmse=-6.409904 dB (best=-6.409963)
i=6000   nmse=-6.410212 dB (best=-6.410276)
i=7000   nmse=-6.410766 dB (best=-6.410766)
i=8000   nmse=-6.410609 dB (best=-6.410862)
i=9000   nmse=-6.410803 dB (best=-6.410868)
i=10000  nmse=-6.410927 dB (best=-6.411133)
i=11000  nmse=-6.411018 dB (best=-6.411149)
i=12000  nmse=-6.410999 dB (best=-6.411149)
i=13000  nmse=-6.411237 dB (best=-6.411236)
i=14000  nmse=-6.411270 dB (best=-6.411353)
i=15000  nmse=-6.411462 dB (best=-6.411462)
i=16000  nmse=-6.411371 dB (best=-6.411601)
i=17000  nmse=-6.411231 dB (best=-6.411601)
i=18000  nmse=-6.411111 dB (best=-6.411601)
i=19000  nmse=-6.411225 dB (best=-6.411601)
i=20000  nmse=-6.411146 dB (best=-6.411601)
i=21000  nmse=-6.411213 dB (best=-6.411601)
LISTA T=2 extending lam_1:0
i=0      nmse=2.815641 dB (best=2.815641)
i=1000   nmse=-2.282928 dB (best=-2.282928)
i=2000   nmse=-3.765755 dB (best=-3.765755)
i=3000   nmse=-4.176616 dB (best=-4.176616)
i=4000   nmse=-4.251509 dB (best=-4.251509)
i=5000   nmse=-4.255618 dB (best=-4.255820)
i=6000   nmse=-4.255032 dB (best=-4.255820)
i=7000   nmse=-4.254972 dB (best=-4.255820)
i=8000   nmse=-4.254987 dB (best=-4.255820)
i=9000   nmse=-4.254854 dB (best=-4.255820)
i=10000  nmse=-4.255017 dB (best=-4.255820)
LISTA T=2 trainrate=0.5 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-4.255017 dB (best=-4.255017)
i=1000   nmse=-6.825531 dB (best=-6.825531)
i=2000   nmse=-9.571950 dB (best=-9.571950)
i=3000   nmse=-11.000087 dB (best=-11.000088)
i=4000   nmse=-11.156223 dB (best=-11.156223)
i=5000   nmse=-11.234140 dB (best=-11.245314)
i=6000   nmse=-11.302651 dB (best=-11.302651)
i=7000   nmse=-11.337249 dB (best=-11.352076)
i=8000   nmse=-11.366589 dB (best=-11.366589)
i=9000   nmse=-11.374066 dB (best=-11.386651)
i=10000  nmse=-11.384757 dB (best=-11.391203)
i=11000  nmse=-11.378998 dB (best=-11.398867)
i=12000  nmse=-11.392558 dB (best=-11.399878)
i=13000  nmse=-11.382673 dB (best=-11.406742)
i=14000  nmse=-11.391981 dB (best=-11.406742)
i=15000  nmse=-11.405396 dB (best=-11.408226)
i=16000  nmse=-11.389142 dB (best=-11.413436)
i=17000  nmse=-11.393719 dB (best=-11.413436)
i=18000  nmse=-11.389047 dB (best=-11.413436)
i=19000  nmse=-11.394378 dB (best=-11.413436)
i=20000  nmse=-11.399170 dB (best=-11.413436)
i=21000  nmse=-11.390210 dB (best=-11.413436)
LISTA T=2 trainrate=0.1 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-11.390210 dB (best=-11.390210)
i=1000   nmse=-11.481429 dB (best=-11.483165)
i=2000   nmse=-11.487813 dB (best=-11.489386)
i=3000   nmse=-11.486307 dB (best=-11.493517)
i=4000   nmse=-11.489860 dB (best=-11.493517)
i=5000   nmse=-11.490422 dB (best=-11.493688)
i=6000   nmse=-11.488150 dB (best=-11.493688)
i=7000   nmse=-11.479529 dB (best=-11.493688)
i=8000   nmse=-11.485313 dB (best=-11.493688)
i=9000   nmse=-11.486683 dB (best=-11.493688)
i=10000  nmse=-11.490084 dB (best=-11.495165)
i=11000  nmse=-11.479270 dB (best=-11.495165)
i=12000  nmse=-11.490300 dB (best=-11.495165)
i=13000  nmse=-11.491095 dB (best=-11.496396)
i=14000  nmse=-11.488986 dB (best=-11.496396)
i=15000  nmse=-11.492862 dB (best=-11.496396)
i=16000  nmse=-11.485879 dB (best=-11.496396)
i=17000  nmse=-11.489718 dB (best=-11.496396)
i=18000  nmse=-11.489605 dB (best=-11.496396)
LISTA T=2 trainrate=0.01 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-11.489605 dB (best=-11.489605)
i=1000   nmse=-11.500024 dB (best=-11.500023)
i=2000   nmse=-11.503768 dB (best=-11.503768)
i=3000   nmse=-11.505711 dB (best=-11.506657)
i=4000   nmse=-11.506758 dB (best=-11.506853)
i=5000   nmse=-11.507987 dB (best=-11.508021)
i=6000   nmse=-11.509105 dB (best=-11.509460)
i=7000   nmse=-11.508684 dB (best=-11.509460)
i=8000   nmse=-11.508490 dB (best=-11.509461)
i=9000   nmse=-11.508588 dB (best=-11.509634)
i=10000  nmse=-11.508828 dB (best=-11.509634)
i=11000  nmse=-11.508963 dB (best=-11.509634)
i=12000  nmse=-11.509893 dB (best=-11.509893)
i=13000  nmse=-11.511170 dB (best=-11.511170)
i=14000  nmse=-11.509981 dB (best=-11.511439)
i=15000  nmse=-11.509600 dB (best=-11.511439)
i=16000  nmse=-11.509181 dB (best=-11.511439)
i=17000  nmse=-11.508384 dB (best=-11.511439)
i=18000  nmse=-11.507735 dB (best=-11.511439)
i=19000  nmse=-11.508274 dB (best=-11.511439)
LISTA T=3 extending lam_2:0
i=0      nmse=-5.825481 dB (best=-5.825482)
i=1000   nmse=-9.813716 dB (best=-9.813716)
i=2000   nmse=-9.816023 dB (best=-9.816034)
i=3000   nmse=-9.816021 dB (best=-9.816034)
i=4000   nmse=-9.816017 dB (best=-9.816034)
i=5000   nmse=-9.816024 dB (best=-9.816034)
i=6000   nmse=-9.816030 dB (best=-9.816034)
i=7000   nmse=-9.816018 dB (best=-9.816034)
LISTA T=3 trainrate=0.5 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-9.816018 dB (best=-9.816018)
i=1000   nmse=-14.083040 dB (best=-14.089805)
i=2000   nmse=-14.262290 dB (best=-14.263415)
i=3000   nmse=-14.351339 dB (best=-14.352772)
i=4000   nmse=-14.367601 dB (best=-14.383436)
i=5000   nmse=-14.394972 dB (best=-14.405580)
i=6000   nmse=-14.408193 dB (best=-14.410375)
i=7000   nmse=-14.398428 dB (best=-14.419014)
i=8000   nmse=-14.422532 dB (best=-14.427329)
i=9000   nmse=-14.389062 dB (best=-14.427329)
i=10000  nmse=-14.394209 dB (best=-14.427329)
i=11000  nmse=-14.410884 dB (best=-14.427329)
i=12000  nmse=-14.410186 dB (best=-14.427329)
i=13000  nmse=-14.416312 dB (best=-14.429248)
i=14000  nmse=-14.417455 dB (best=-14.429757)
i=15000  nmse=-14.401765 dB (best=-14.429757)
i=16000  nmse=-14.398332 dB (best=-14.429757)
i=17000  nmse=-14.400302 dB (best=-14.429757)
i=18000  nmse=-14.390738 dB (best=-14.429757)
i=19000  nmse=-14.424274 dB (best=-14.435135)
i=20000  nmse=-14.407024 dB (best=-14.435135)
i=21000  nmse=-14.415981 dB (best=-14.435135)
i=22000  nmse=-14.369830 dB (best=-14.435135)
i=23000  nmse=-14.398458 dB (best=-14.435135)
i=24000  nmse=-14.409600 dB (best=-14.435135)
LISTA T=3 trainrate=0.1 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-14.409600 dB (best=-14.409601)
i=1000   nmse=-14.601861 dB (best=-14.610094)
i=2000   nmse=-14.612112 dB (best=-14.619740)
i=3000   nmse=-14.607580 dB (best=-14.619740)
i=4000   nmse=-14.605111 dB (best=-14.619740)
i=5000   nmse=-14.601336 dB (best=-14.619740)
i=6000   nmse=-14.606397 dB (best=-14.619740)
i=7000   nmse=-14.605229 dB (best=-14.619740)
LISTA T=3 trainrate=0.01 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-14.605229 dB (best=-14.605228)
i=1000   nmse=-14.637463 dB (best=-14.637464)
i=2000   nmse=-14.643615 dB (best=-14.644059)
i=3000   nmse=-14.650302 dB (best=-14.650302)
i=4000   nmse=-14.650685 dB (best=-14.651823)
i=5000   nmse=-14.652174 dB (best=-14.652901)
i=6000   nmse=-14.650363 dB (best=-14.652901)
i=7000   nmse=-14.654099 dB (best=-14.654294)
i=8000   nmse=-14.654527 dB (best=-14.655034)
i=9000   nmse=-14.654653 dB (best=-14.655034)
i=10000  nmse=-14.655502 dB (best=-14.655501)
i=11000  nmse=-14.656656 dB (best=-14.656891)
i=12000  nmse=-14.655936 dB (best=-14.657195)
i=13000  nmse=-14.656190 dB (best=-14.657195)
i=14000  nmse=-14.657965 dB (best=-14.658177)
i=15000  nmse=-14.656084 dB (best=-14.658351)
i=16000  nmse=-14.655294 dB (best=-14.658351)
i=17000  nmse=-14.653323 dB (best=-14.658351)
i=18000  nmse=-14.654233 dB (best=-14.658351)
i=19000  nmse=-14.653833 dB (best=-14.658351)
i=20000  nmse=-14.653200 dB (best=-14.658351)
LISTA T=4 extending lam_3:0
i=0      nmse=-12.571206 dB (best=-12.571206)
i=1000   nmse=-13.083762 dB (best=-13.083871)
i=2000   nmse=-13.083782 dB (best=-13.083886)
i=3000   nmse=-13.083868 dB (best=-13.083891)
i=4000   nmse=-13.083812 dB (best=-13.083891)
i=5000   nmse=-13.083787 dB (best=-13.083891)
i=6000   nmse=-13.083525 dB (best=-13.083891)
i=7000   nmse=-13.083661 dB (best=-13.083891)
i=8000   nmse=-13.083868 dB (best=-13.083891)
i=9000   nmse=-13.083730 dB (best=-13.083891)
i=10000  nmse=-13.083568 dB (best=-13.083891)
i=11000  nmse=-13.083565 dB (best=-13.083891)
i=12000  nmse=-13.083783 dB (best=-13.083891)
LISTA T=4 trainrate=0.5 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-13.083783 dB (best=-13.083784)
i=1000   nmse=-16.955930 dB (best=-16.968576)
i=2000   nmse=-16.970395 dB (best=-17.004757)
i=3000   nmse=-17.067116 dB (best=-17.067117)
i=4000   nmse=-16.994594 dB (best=-17.067117)
i=5000   nmse=-17.026620 dB (best=-17.067117)
i=6000   nmse=-16.962693 dB (best=-17.067117)
i=7000   nmse=-17.021729 dB (best=-17.067117)
i=8000   nmse=-16.997473 dB (best=-17.067117)
i=9000   nmse=-16.984513 dB (best=-17.067117)
LISTA T=4 trainrate=0.1 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-16.984513 dB (best=-16.984513)
i=1000   nmse=-17.330179 dB (best=-17.346174)
i=2000   nmse=-17.339813 dB (best=-17.351166)
i=3000   nmse=-17.329377 dB (best=-17.351166)
i=4000   nmse=-17.339823 dB (best=-17.357078)
i=5000   nmse=-17.334964 dB (best=-17.357865)
i=6000   nmse=-17.330539 dB (best=-17.357865)
i=7000   nmse=-17.342944 dB (best=-17.357865)
i=8000   nmse=-17.340785 dB (best=-17.357865)
i=9000   nmse=-17.349666 dB (best=-17.357865)
i=10000  nmse=-17.338438 dB (best=-17.357865)
LISTA T=4 trainrate=0.01 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-17.338438 dB (best=-17.338438)
i=1000   nmse=-17.400519 dB (best=-17.400676)
i=2000   nmse=-17.411553 dB (best=-17.412855)
i=3000   nmse=-17.417825 dB (best=-17.417826)
i=4000   nmse=-17.418836 dB (best=-17.420176)
i=5000   nmse=-17.416502 dB (best=-17.420427)
i=6000   nmse=-17.423451 dB (best=-17.424679)
i=7000   nmse=-17.416817 dB (best=-17.425653)
i=8000   nmse=-17.422128 dB (best=-17.425653)
i=9000   nmse=-17.423766 dB (best=-17.425653)
i=10000  nmse=-17.420039 dB (best=-17.425653)
i=11000  nmse=-17.419932 dB (best=-17.425653)
i=12000  nmse=-17.420735 dB (best=-17.425653)
LISTA T=5 extending lam_4:0
i=0      nmse=-17.475288 dB (best=-17.475288)
i=1000   nmse=-17.536618 dB (best=-17.536687)
i=2000   nmse=-17.536635 dB (best=-17.536689)
i=3000   nmse=-17.536681 dB (best=-17.536689)
i=4000   nmse=-17.536510 dB (best=-17.536689)
i=5000   nmse=-17.536455 dB (best=-17.536689)
i=6000   nmse=-17.536511 dB (best=-17.536689)
i=7000   nmse=-17.536607 dB (best=-17.536689)
LISTA T=5 trainrate=0.5 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-17.536607 dB (best=-17.536607)
i=1000   nmse=-19.414814 dB (best=-19.505329)
i=2000   nmse=-19.453874 dB (best=-19.505329)
i=3000   nmse=-19.438593 dB (best=-19.513495)
i=4000   nmse=-19.438152 dB (best=-19.513495)
i=5000   nmse=-19.446702 dB (best=-19.513495)
i=6000   nmse=-19.437819 dB (best=-19.513495)
i=7000   nmse=-19.447709 dB (best=-19.513495)
i=8000   nmse=-19.426291 dB (best=-19.513495)
LISTA T=5 trainrate=0.1 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-19.426291 dB (best=-19.426290)
i=1000   nmse=-19.999832 dB (best=-19.999831)
i=2000   nmse=-19.985374 dB (best=-20.012554)
i=3000   nmse=-19.991257 dB (best=-20.027865)
i=4000   nmse=-19.999701 dB (best=-20.027865)
i=5000   nmse=-20.001900 dB (best=-20.027865)
i=6000   nmse=-20.011096 dB (best=-20.027865)
i=7000   nmse=-20.007813 dB (best=-20.027927)
i=8000   nmse=-20.001535 dB (best=-20.031084)
i=9000   nmse=-20.001183 dB (best=-20.031084)
i=10000  nmse=-19.999759 dB (best=-20.031084)
i=11000  nmse=-19.987308 dB (best=-20.031084)
i=12000  nmse=-19.998591 dB (best=-20.031084)
i=13000  nmse=-19.997168 dB (best=-20.031084)
LISTA T=5 trainrate=0.01 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-19.997168 dB (best=-19.997168)
i=1000   nmse=-20.119538 dB (best=-20.119538)
i=2000   nmse=-20.133333 dB (best=-20.134353)
i=3000   nmse=-20.142293 dB (best=-20.143895)
i=4000   nmse=-20.136995 dB (best=-20.143895)
i=5000   nmse=-20.135386 dB (best=-20.143895)
i=6000   nmse=-20.133083 dB (best=-20.143895)
i=7000   nmse=-20.136383 dB (best=-20.143895)
i=8000   nmse=-20.134094 dB (best=-20.143895)
LISTA T=6 extending lam_5:0
i=0      nmse=-18.395469 dB (best=-18.395470)
i=1000   nmse=-18.997887 dB (best=-18.997951)
i=2000   nmse=-18.997630 dB (best=-18.997961)
i=3000   nmse=-18.997049 dB (best=-18.997962)
i=4000   nmse=-18.997921 dB (best=-18.997962)
i=5000   nmse=-18.997855 dB (best=-18.997962)
i=6000   nmse=-18.997881 dB (best=-18.997962)
i=7000   nmse=-18.997895 dB (best=-18.997962)
i=8000   nmse=-18.997780 dB (best=-18.997962)
LISTA T=6 trainrate=0.5 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-18.997780 dB (best=-18.997780)
i=1000   nmse=-21.296265 dB (best=-21.395632)
i=2000   nmse=-21.299195 dB (best=-21.395632)
i=3000   nmse=-21.218281 dB (best=-21.395632)
i=4000   nmse=-21.262178 dB (best=-21.395632)
i=5000   nmse=-21.250138 dB (best=-21.395632)
i=6000   nmse=-21.314480 dB (best=-21.395632)
LISTA T=6 trainrate=0.1 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-21.314480 dB (best=-21.314480)
i=1000   nmse=-22.063572 dB (best=-22.092929)
i=2000   nmse=-22.087989 dB (best=-22.109940)
i=3000   nmse=-22.076800 dB (best=-22.109940)
i=4000   nmse=-22.065811 dB (best=-22.109940)
i=5000   nmse=-22.068307 dB (best=-22.109940)
i=6000   nmse=-22.068260 dB (best=-22.109940)
i=7000   nmse=-22.082293 dB (best=-22.109940)
LISTA T=6 trainrate=0.01 fine tuning all B_0:0,S_0:0,lam_0:0,lam_1:0,lam_2:0,lam_3:0,lam_4:0,lam_5:0
i=0      nmse=-22.082293 dB (best=-22.082293)
i=1000   nmse=-22.246459 dB (best=-22.246458)
i=2000   nmse=-22.255030 dB (best=-22.261591)
i=3000   nmse=-22.273688 dB (best=-22.274738)
i=4000   nmse=-22.271194 dB (best=-22.275748)
i=5000   nmse=-22.270195 dB (best=-22.275748)
i=6000   nmse=-22.273598 dB (best=-22.279479)
i=7000   nmse=-22.267370 dB (best=-22.279479)
i=8000   nmse=-22.274561 dB (best=-22.279479)
i=9000   nmse=-22.266262 dB (best=-22.279479)
i=10000  nmse=-22.258878 dB (best=-22.279479)
i=11000  nmse=-22.268763 dB (best=-22.279479)
```
## [LAMP.py](LAMP.py)

Example of Learned AMP (LAMP) with a variety of shrinkage functions.

## [LVAMP.py](LVAMP.py)

Example of Learned Vector AMP (LVAMP).

