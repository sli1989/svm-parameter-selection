Support Vector Machines have emerged as an important learning technique for solving classification and regression problems in various fields.  However, poor choice of penalty parameter and kernel parameters can dramatically decrease the performance of SVR. Naturally, without the inefficient search strategies and long searching time in heuristics algorithms, meta-heuristics have been introduced as problem-independent technique to obtain an acceptable optimum in a wide range of problems.

So this program aims to provide simulation codes of meta-heuristics for SVM parameter selection. 

## How to use

The basic GS-SVR and GA-SVR demo can be found in `matlab-implement` folder in [Libsvm-FarutoUltimate-Version](https://github.com/faruto/Libsvm-FarutoUltimate-Version).

```
[bestCVmse,bestc,bestg] = SVMcgForRegress(TrainL,Train,-8,8,-8,8,5,0.4,0.4)
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];
```

Many Metaheuristics code can be found in [yarpiz.com](http://yarpiz.com/category/metaheuristics) or [matlab fileexchange](https://www.mathworks.com/matlabcentral/fileexchange).The more meta-heuristics based algorithm can be infered by `ABC_SVM`.

To be continued.

## Research papers

- Sai Li, Huajing Fang, and Xiaoyong Liu. "Parameter Optimization of Support Vector Regression Based on Sine Cosine Algorithm." Expert Systems with Applications (2017). (SCI，Impact Factor：3.928，doi:10.1016/j.eswa.2017.08.038)
- Sai Li, Huajing Fang. "A WOA-based algorithm for parameter optimization of support vector regression and its application to condition prognostics." Control Conference (CCC), 2017 36th Chinese. IEEE, 2017.  (EI Compendex，DOI: 10.23919/ChiCC.2017.8028516)