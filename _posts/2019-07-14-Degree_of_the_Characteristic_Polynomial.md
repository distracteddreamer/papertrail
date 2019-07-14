---
layout: post
title:  "Degree of the Characteristic Polynomial"
date:   2019-07-14 11:51:32 +0100
categories: jekyll update
---

We would like to show that $p_A(\lambda) = \lvert A - \lambda I_n\rvert$ is a polynomial of degree $n$ in $\lambda$.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sy
%matplotlib inline
```

Some functions to plot a matrix


```python
def plot_mat(A, annot):
    n = A.shape[0]
    plt.figure(figsize=(n, n))
    sns.heatmap(A, 
                annot=annot, 
                annot_kws={'fontsize':16, 'fontweight':'bold'}, fmt = '', 
                xticklabels=False, 
                yticklabels=False,
                cbar=False, 
                square=True)
    
def get_annot(A):
    annot = A.astype('str')
    diag = np.diag(annot)
    annot[np.diag_indices_from(annot)] = [(a * (int(a) != 0)) + 
                                          ' - $\lambda$' for a in diag]
    return annot

def submat(A, i, j):
    A1 = np.concatenate([A[:i, :j], A[:i, (j+1):]], axis=-1)
    A2 = np.concatenate([A[(i+1):, :j], A[(i+1):, (j+1):]], axis=-1)
    return np.concatenate([A1, A2], axis=0)

def submat_sym(A, i, j):
    A1 = A[:i, :j].row_join(A[:i, (j+1):])
    A2 = A[(i+1):, :j].row_join(A[(i+1):, (j+1):])
    return A1.col_join(A2)


```

We would like to prove that the characteristic polynomial of a matrix has order $n$.

Let us first note that we are concerned with $n \times n$ matrices for which:
- There are no more than $n$ entries of the form $a - \lambda$ for a constant $a$
- All other entries constant.
- The $a - \lambda$ elements appear no more than once per column and per row.

An $n \times n$ matrix such as the following random $5 x 5$ matrix with $\lambda$ subtracted from its diagonal entries, whose determinant is obtained to find its eigenvalues is such an example.


```python
n = 5
np.random.seed(27)
A = np.random.permutation(np.arange(-10, 15)).reshape((5, 5))
annot = get_annot(A)
plot_mat(A, annot)
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_5_0.png)



```python
A_sym = sy.Matrix(A) - sy.Symbol('\lambda') * sy.eye(5)
A_sym
```




$$\displaystyle \left[\begin{matrix}10 - \lambda & 13 & -3 & 8 & 1\\2 & - \lambda - 1 & 11 & -10 & 5\\-7 & -4 & - \lambda - 6 & 12 & 14\\4 & -9 & 0 & - \lambda - 8 & 7\\3 & -5 & 6 & -2 & 9 - \lambda\end{matrix}\right]$$




```python
sy.det(A_sym)
```




$\displaystyle - \lambda^{5} + 4 \lambda^{4} + 306 \lambda^{3} - 3679 \lambda^{2} + 5942 \lambda + 192420$



The submatrix obtaining by omitting row $i$, column $j$ is another example. When $i = j$, there are exactly $n=4$ such entries. 


```python
plot_mat(submat(A, 1, 1), submat(annot, 1, 1))
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_9_0.png)


When $i \neq j$ there are strictly fewer than $n=4$ entries.


```python
submat_sym(A_sym, 1, 1)
```




$$\displaystyle \left[\begin{matrix}10 - \lambda & -3 & 8 & 1\\-7 & - \lambda - 6 & 12 & 14\\4 & 0 & - \lambda - 8 & 7\\3 & 6 & -2 & 9 - \lambda\end{matrix}\right]$$




```python
sy.det(submat_sym(A_sym, 1, 1))
```




$\displaystyle \lambda^{4} - 5 \lambda^{3} - 254 \lambda^{2} + 183 \lambda + 22302$




```python
plot_mat(submat(A, 3, 1), submat(annot, 3, 1))
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_13_0.png)



```python
submat_sym(A_sym, 3, 1)
```




$$\displaystyle \left[\begin{matrix}10 - \lambda & -3 & 8 & 1\\2 & 11 & -10 & 5\\-7 & - \lambda - 6 & 12 & 14\\3 & 6 & -2 & 9 - \lambda\end{matrix}\right]$$




```python
sy.det(submat_sym(A_sym, 3, 1))
```




$\displaystyle - 10 \lambda^{3} + 268 \lambda^{2} - 1716 \lambda + 1206$



The proof is by induction:
- First note that we will only be concerned with matrices as described above i.e. no more than one $a - \lambda$  type term for every column and row.
- We prove a more general case that for such a matrix:
    - If it has $n$ $a - \lambda$ type entries, the determinant is degree $n$ in $\lambda$.
    - If degree $n$ in $\lambda$, then coeffient of $\lambda^n$ in the polynomial is $\pm1$
    - If fewer than $n$ such entries the degree is $< n$.

**Base case**
- Note that for a $1 \times 1$ matrix which is a scalar, the determinant is degree $1$ in $\lambda$ if the scalar is of the form $a - \lambda$  
- So for the base case $p_A(\lambda)$ is degree $n=1$ with coefficient $-1$ if there are $n$ such entries else it is degree $<n$.

**Induction step**
- Assume it is true for a $k \times k$ matrix:
    1. Consider a $k + 1 \times k + 1$ matrix with $k + 1$ such entries.
        - All rows will have an $a - \lambda$  term.
        - Take the determinant with respect to row 1.
        - Suppose row 1 has such an entry at position $t$
        - If you eliminate row 1, column $t$ the resulting submatrix will have $k$ such entries since the remaining entries will have to occur in other columns and rows due to the restriction of no more than one such entry per column and row.
        - But all other submatrices 
        - So the determinant of that submatrix will a polynomial of degree $k$.
        - Since $A_{1t}$ is a polynomial degree $1$, $A_{1t}$  times that will be a polynomial of degree $k + 1$.
        - This term alone will contain $\lambda^{k + 1}$ since all other submatrix determinants will be polynomials of degree less than $k$ 
        - Assuming that the coefficient of $\lambda^k$ the determinant polynomial is $\pm1$ since now it will be multiplied by $a - \lambda$ the coefficient of $\lambda^{k+1}$ will remain $\pm1$

# Case 1


```python
shuffle = [2, 1, 3, 0, 4]
A_perm = A[shuffle]
annot_perm = annot[shuffle]
```


```python
plot_mat(A_perm, annot_perm)
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_19_0.png)


coefficient = -1, degree = 5


```python
A_perm_sym = A_sym[shuffle, :]
sy.det(A_perm_sym)
```




$\displaystyle - \lambda^{5} + 4 \lambda^{4} + 306 \lambda^{3} - 3679 \lambda^{2} + 5942 \lambda + 192420$




```python
plot_mat(submat(A_perm, 0, 2), submat(annot_perm, 0, 2))
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_22_0.png)


coefficient = 1, degree = 4


```python
sy.det(submat_sym(A_perm_sym, 0, 2))
```




$\displaystyle \lambda^{4} - 10 \lambda^{3} - 185 \lambda^{2} + 2326 \lambda - 6840$




```python
plot_mat(submat(A_perm, 2, 1), submat(annot_perm, 2, 1))
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_25_0.png)


degree < 4


```python
sy.det(submat_sym(A_perm_sym, 2, 1))
```




$\displaystyle 10 \lambda^{3} - 268 \lambda^{2} + 1716 \lambda - 1206$



2. Consider the other case of a $k + 1 \times k + 1$ matrix with $p < k + 1$ such entries.
    - Then at least one row of $A$ does not have an $a - \lambda$ term.
    - Let us take the determinant with respect to this row which means each submatrix determinant will multiplied by a constant
    - Each submatrix has $p$ or fewer $a - \lambda$ terms/
    - If the submatrix has $p$ $a - \lambda$ terms, its determinant is degree $p$ otherwise its determinant is degree less than $p$ (by the induction assumption).
    - Since in the determinant sum each submatrix determinant will be multiplied by a constant its degree will not change.
    - So the determinant of the $k + 1 \times k + 1$ matrix will be a sum of polynomials of degree $p < k + 1$ and thus will be of degree strictly less than $k +1$
    
This completes the proof. The characteristic polynomial is a special case of such a matrix where all the $n$ diagonal entries are of the form $a - \lambda$ so it is a polynomial of degree $n$ with $\lambda^n$ having a coefficient $\pm1$.

# Case 2


```python
A_skip31 = submat(A, 3, 1)
A_skip31_sym = submat_sym(A_sym, 3, 1)
annot_skip31 = submat(annot, 3, 1)
```


```python
plot_mat(A_skip31, annot_skip31)
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_31_0.png)


degree < 4


```python
sy.det(A_skip31_sym)
```




$\displaystyle - 10 \lambda^{3} + 268 \lambda^{2} - 1716 \lambda + 1206$




```python
A_skip31_skip12 = submat(A_skip31, 1, 2)
A_skip31_skip12_sym = submat_sym(A_skip31_sym, 1, 2)
annot_skip31_skip12 = submat(annot_skip31, 1, 2)
```


```python
plot_mat(A_skip31_skip12, annot_skip31_skip12)
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_35_0.png)


coefficient = -1, degree = 3


```python
sy.det(A_skip31_skip12_sym)
```




$\displaystyle 108 \lambda + \left(9 - \lambda\right) \left(10 - \lambda\right) \left(- \lambda - 6\right) - 1179$




```python
A_skip31_skip33 = submat(A_skip31, 3, 3)
A_skip31_skip33_sym = submat_sym(A_skip31_sym, 3, 3)
annot_skip31_skip33 = submat(annot_skip31, 3, 3)
```


```python
plot_mat(A_skip31_skip33, annot_skip31_skip33)
```


![png]({{ site.baseurl }}/assets/Degree_of_the_Characteristic_Polynomial_files/Degree_of_the_Characteristic_Polynomial_39_0.png)


degree < 3


```python
sy.det(A_skip31_skip33_sym) 
```




$\displaystyle - 148 \lambda - \left(- \lambda - 6\right) \left(10 \lambda - 100\right) + 1702$


