# Overview
## Notation
response/target = *Y*
feature/input/predictor = *X_n*
Y = f(X) + e (error)

model importance?
- prediction of Y
- weight of X_n on *explaining* Y
- depending on complexity, how X_n affect Y

f(x) = E(Y| X=x) is called *regression function*, f(x) interpreting the
*expecting value of Y given X=x*.

### ideal/optimal predictor of Y:
  the function that minimize E[ (Y-g(x))^2 | X=x ] over all functions g at all points
  X = x; (basically minize the sum of mean-squared prediction error)

### irreducible error:
  e = Y - f(x)

--> what if there's no data points to average
f(x) = Ave(Y | X ~ N(x) ), where N=*some neighborhood function*,
  *local averaging function / nearest neighbor*
This works well with smaller feature and large N (sample).

--> higher dimensionality ==> loss of localiy
we need small variance for NN, so need more data samples, if capture 10% data point;
in higher dimention, to include 10% may have largely diverged feature value.

--> to deal with this, introduce *parametric and structured models*
because it doesnt use any local/neighbor knowledge, its almost never correct, but it
serves as a good approximation.


# Model selection & Bias-variance
MSE = mean square error (on *test* vs. *train*)
E(y - f'(x)) = Var(f(x)) + [Bias(f(x))] ^2 + Var(e)
adding the *variance curve* with *bias* curve equals roughly the *MSE curve* which
includes the irreducible error Var(e);

# Classification Problem
Conditional class probabilities at x:
p_k(x) = Pr(Y=k | X=x), where k = 1 ... K, these are based on observational data.

Bayes optimal classifier:
  C(X) = j if p_j(X) = max{ p_1(x), ... p_k(x),}

##two approaches:
- SVM build structured models for *C(x)*
  - just interested in building a classifier;
- Logistic regression (additive models, etc) build structured model for
 representing *p_k(x)*;
  - interested in probabilities themselves;

## KNN
in the example;
- KNN-1: find the 1 nearest neighbor x_i, classify to y_i;
decision boundary = points where p1(x_i) = p2(x_i);
"overfitting"
- KNN-100: underfitting

very powerful technique, for Classification Problem;
