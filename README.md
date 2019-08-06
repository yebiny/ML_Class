
# Class Overview
## 02 Gradient descent
## 03 Regression  
### 
- Regression is one of the major tasks in machine learning
- Idea: given some pieces of data, can you predict some dependent variables
- Examples
  - Can you predict the sale price of a house given location, size,
    no. of rooms, etc.
  - Given response of a calorimeter (measured in ADC counts), can you
    find the energy of the incoming particle
- Often the question is not so much to predict a quantity, but ask if
  there's a correlation between variables
  - Ex: Does the number of years in school impact your future salary
  - More concretely: how much of the variance in salary is explained
    by years of education
- Much of the statistics in regression is to say how significant the
  correlation between the variables is
- We'll contain ourselves mostly to the question of prediction, in
  particular /parametric regression/

## 04 Multiple regression
## 05 Logistic regression
## 06 Multinomial regression
## 07 Neural networks
### Introduction
* Some very simple examples for simple logistic regression
- Let's think about using logistic regression to approximate some
  simple binary functions
- OR and AND gates
  - OR is 0 (red) if both input are 0, 1 (blue) otherwise
  - AND is 1 if both inputs are 1, 0 otherwise
- Can we find logistic function approximations for this?
  - That is, \(f(x_1, x_2)\) returns approximately 1 or 0 at the indicated points \pause
- Yes! Take the projection perpendicular to the line \pause
- and have the logistic turn on at the line
  - e.g. \(f(x_1, x_2) = \sigma(2 x_1 + 2 x_2 - 1)\) for OR, \(f(x_1, x_2) = \sigma(2 x_1 + 2 x_2 - 3)\) for AND [\sigma is our logistic function]


## 08 Backpropagation 
## 09 Decision trees
## 10 Training trees
  
