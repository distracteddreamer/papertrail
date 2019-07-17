---
layout: post
title:  "Notes on eigenvalues and eigenvectors, Part 1"
date:   2019-07-06 11:51:32 +0100
categories: jekyll update
---
These are my study notes for [Eigenvalues and Eigenvectors](http://linear.ups.edu/html/section-EE.html) from [A First Course in Linear Algebra](http://linear.ups.edu/html/fcla.html) by Robert A. Beezer. All mistakes are my own. Please see the text for detailed explanations and examples. 

#  What are eigenvalues and eigenvectors?
- For a square matrix $A$ there exist vectors $x$ such the transformation $A\mathbf{x}$ is a simple scaling $\lambda\mathbf{x}$.
- An eigenvalue, eigenvector pair for $A$ is a scalar $\lambda$ and a non -zero vector $\mathbf{x}$ such that:
    $$A\mathbf{x}=\lambda\mathbf{x}$$.
- Every square matrix has an eigenvalue.

# Characteristic polynomial
- Let $A$ be an $n \times n$ matrix
- $p_A(\lambda) = \det(A - \lambda I_n)$
- The solutions of $p_A(\lambda) = 0$ are the eigenvalues of $A$.
- Proof that $\lambda$ is an eigenvalue if and only if $p_A(\lambda) = 0$:
    
    $$\begin{align} &\lambda \text{ eigenvalue of } A  
   \\&\Leftrightarrow \exists \mathbf{x} \neq \mathbf{0} \text{ s.t. } A\mathbf{x} = \lambda\mathbf{x} 
    \\&\Leftrightarrow  (A - \lambda I_n)\mathbf{x} = \mathbf{0}  
    \\&\Leftrightarrow (A - \lambda I_n) \text{ }\text{ is singular}
   	\\&\Leftrightarrow \det(A - \lambda I_n) = 0
    \\&\Leftrightarrow p_A(\lambda) = 0 \end{align}$$
- The eigenvectors corresponding to each solution are the set of non-zero vectors in the null space of $(A - \lambda I_n)\mathbf{x}$ (see proof below).

# Eigenspace of a matrix
- $\xi_A(\lambda)$ is the set of eigenvectors of $A$ corresponding to the eigenvalue $\lambda$ plus the zero vector $\mathbf{0}$.
- It is a subspace because any linear combination of vectors in $\xi_A(\lambda)$  is also in $\xi_A(\lambda)$ :

    $$\mathbf{u}, \mathbf{v} \in \xi_A(\lambda), \mathbf{w} 
    = \alpha\mathbf{u} + \beta\mathbf{v}
    \\ A\mathbf{w} = A(\alpha\mathbf{u} + \beta\mathbf{v})
    = \alpha A\mathbf{u} + \beta A\mathbf{v}
    = \alpha \lambda\mathbf{u} + \beta \lambda\mathbf{v} 
    = \lambda(\alpha\mathbf{u} + \beta\mathbf{v})
    = \lambda \mathbf{w}$$

    $\implies \mathbf{w}$ is eigenvector of $A$ with eigenvalue $\lambda$ or $\mathbf{w} = \mathbf{0}$, both of of which are in the set $\xi_A(\lambda)$ so $\xi_A(\lambda)$ is a subspace.

- It is the nullspace of $A - \lambda I_n$ i.e. $\xi_A(\lambda) = \mathcal{N}(A - \lambda I_n)$:

    $$\mathbf{x} \in \xi_A(\lambda)
    \Leftrightarrow A\mathbf{x} = \lambda \mathbf{x}, \mathbf{x} \neq 0 \text{ }\text{or}\text{ } \mathbf{x} = \mathbf{0}
    \Leftrightarrow (A - \lambda I_n)\mathbf{x} = \mathbf{0}
    \Leftrightarrow \mathbf{x} \in \mathcal{N}(A - \lambda I_n)$$

# Multiplicities of an eigenvalue
- **Algebraic multiplicity** is the highest power $p$ of $(x - \lambda)$ such that $(x - \lambda)^p$ exactly divides $p_A(x)$ i.e. the number of times $\lambda$ is repeated root of the characteristic polynomial. 
- **Geometric multiplicity** is the dimension of the eigenspace of $\lambda$, $\xi_A(\lambda)$.

# References
[http://linear.ups.edu/html/section-EE.html](http://linear.ups.edu/html/section-EE.html)




