<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Dimensionality Reduction](#dimensionality-reduction)
	- [Basic introduction](#basic-introduction)
	- [PCA](#pca)
		- [Optimization objective](#optimization-objective)
		- [How PCA works?](#how-pca-works)
		- [Math behind PCA](#math-behind-pca)
	- [LDA](#lda)
	- [References](#references)

<!-- /TOC -->

# Dimensionality Reduction

## Basic introduction

- _**Why dimensionality reduction?**_
    -  Remove the redundant dimensions.
    -  Only keep the most important dimensions.

- _**How to do dimensionality reductioin**_

  There are two principal algorithms for dimensionality reduction: PAC and LDA.

  ||_PCA_|_LDA_|
  |--|--|--|
  |_optimization objective_|uses the variance of each feature|uses information of classes to find new features to maximize its separability|
  ||unsupervised algorithm|supervised algorithm|

## PCA

### Optimization objective

We expect:

1. the data are spread out across each dimension.
1. the dimensions to be independent.

### How PCA works?

_**Goal**_

1. Find a new set of dimensions such that all the dimensions are orthogonal (and hence linearly independent).
1. Rank these dimensions according to the variance of data along them: more important principle axis occurs first.
    - more important = more variance/more spread out data

_**Process of PCA**_

1. Center input data matrix $X$.
1. Calculate the covariance matrix $X$ of data points.
1. Calculate eigenvectors and corresponding eigenvalues.
1. Sort the eigenvectors according to their eigenvalues in decreasing order.
1. Choose first $k$ , and that will be the new $k$ dimensions.
1. Transform the original $n$ dimensional data points into $k$ dimensions.

### Math behind PCA

## LDA

## References

1. [Dimensionality Reduction: Does PCA really improve the outcome of classification?](https://meigarom.github.io/blog/pca.html)
1. [Understanding Principal Component Analysis](https://medium.com/@aptrishu/understanding-principle-component-analysis-e32be0253ef0)
