---
title:  "Implementing neural network from scratch!"
header:
  teaser: "https://farm5.staticflickr.com/4076/4940499208_b79b77fb0a_z.jpg"
categories: 
  - Neural network
tags:
  - Neural network
  - Machine learning
mathjax: true
---

In this post we will try to understand how a neural network works by implementing it completely from scratch. You'll get to understand what really goes on behind the scene of this network. At the end of this post, you'll not only understand the concept of a neural network, but you will be able to implement one yourself completely from scratch. We are going to achieve this by looking at a practical example as a fun project.
I'm going to try as much as possible to breakdown some of the concepts as much as possible. You don't need to be a maths genius to effectively understand these concepts. 

In a neural network, we have the input layer, the hidden layer and the output layer. The input layer consists of features known as the input features. In our example, they are represented as $${x_1}$$, $${x_2}$$ and $${x_3}$$ which are fed into the hidden layer. The hidden layer on the other hand consists of various nodes. It is termed hidden layer because the true values for the nodes in the middle are not observe in the training set. In other words, we don't see what they should be in the training set. The output layer is responsible for generating the predicted value $${\hat{y}}$$.

We will introduce a concept called the $$\textbf{activations}$$. These are the values that different layers of the network are passing on to the subsequent layers. This means that the input layer passes on the value $$X$$ to the hidden layer. This is called the $$ \textbf{activations}$$ of the input layer denoted as $$a^{[0]}$$. The hidden layer will inturn generate some form of activations denoted as $${a}^{[1]}$$. So the first node in the hidden layer will generate a value $${a}_{1}^{[1]}$$, the second node will generate the value $${a}_{1}^{[1]}$$ and so on till it gets to $${a}_{n}^{[2]}$$ which indicates the last node of that hidden layer. Finally, the output layer will generate a value $${a}^{[2]}$$ which is just a real number. So $${\hat{y}}$$, our target out will take the value of $${a}^{[2]}$$,  $${a}^{[2]} = {\hat{y}} $$.

In neural network representation, the input layer is not usually counted. For example, our network representation above is called a 2-layer neural network. The hidden and the output layers have some other parameters they are associated with, these parameters  are $$W$$ and $$b$$. In our example $$Z=WX + b$$, where $$W$$ is a matrix, $$b$$ is called the bias. Not to confuse you too much at this stage, we will delve into this parameters later in this post.
Now that you have a better understanding of a neural network representation, we are now going to see what these different layers are computing. The first is $$ Z_1^{[1]} = W_1^{[1]T}{X}+b_1^{[1]} $$ and $$a_1^{[1]} = {\sigma}(Z_1^{[1]})$$. For both $$Z$$ and $$a$$, the notation convention is $$a_i^{[l]}$$, where $$[l]$$ denotes the layer number and $$(i)$$ denotes the node in that layer.
For a neural network representation of our example, we will have the following set of equations;

$$Z_1^{[1]} = W_1^{[1]T}{X}+b_1^{[1]} $$, $$a_1^{[1]} = {\sigma}(Z_1^{[1]})$$

$$Z_2^{[1]} = W_2^{[1]T}{X}+b_2^{[1]} $$, $$a_2^{[1]} = {\sigma}(Z_3^{[1]})$$

$$Z_3^{[1]} = W_3^{[1]T}{X}+b_3^{[1]} $$, $$a_3^{[1]} = {\sigma}(Z_3^{[1]})$$

We will start by taking the values of $$Z$$ and vectorize them. First let us take these values and stack them into a matrix.

$$\begin{bmatrix}
	\dots & W_1^{[1]T} & \dots \\
	\dots & W_2^{[1]T} & \dots \\
	\dots & W_3^{[1]T} & \dots 
\end{bmatrix}$$
$$\begin{bmatrix}
	x_1\\
	x_2\\
	x_3
\end{bmatrix}$$
+
$$\begin{bmatrix}
	b_1^{[1]}\\
	b_2^{[1]}\\
	b_3^{[1]}
\end{bmatrix}$$

We can then compute this matrix to have $$Z^{[1]}$$ as follows

$$Z^{[1]} =
\begin{bmatrix}
	W_1^{[1]T} + b_1^{[1]}\\
	W_2^{[1]T} + b_2^{[1]}\\
	W_3^{[1]T} + b_3^{[1]}
\end{bmatrix}$$
=
$$\begin{bmatrix}
	Z_1^{[1]}\\
	Z_2^{[1]}\\
	Z_3^{[1]}
\end{bmatrix}$$

Similarly, we take the elements of $$a^{[l]}$$ and stack them together to obtain our $$a^{[1]}$$;

$$a^{[1]}
\begin{bmatrix}
	a_1^{[1]}\\
	a_2^{[1]}\\
	a_3^{[1]}
\end{bmatrix}$$
= 
$${\sigma}(Z_1^{[1]})$$

where $${\sigma}$$ is the sigmoid function that takes in the four elements of $$Z$$ and applies the sigmoid function elementwise to it. 
From our example, it is worth nothing that the dimensions of $$W$$ is a (3*3) matrix, while that of $$X$$ is a (3X1) matrix and $$b$$ is a (3X1) matrix. 
## Vectorizing across multiple examples
So far what we have been doing is considering a single training example where for a given input wan can predict the output $${\hat{y}}$$. For most practical applications in machine learning, there are always large number of training examples. For the purpose of this post, let us call it $$m$$ training examples. Now for $$m$$ number of training examples, we need to repeat the process of predicting the output $${\hat{y}}$$

For large number of training examples, we repeat each steps as follow
	
for i=1 to m:

$$Z_1^{[1](1)} = W_1^{[1]T}{X^{(1)}}+b_1^{[1]} $$, $$a_1^{[1](1)} = {\sigma}(Z_1^{[1](1)})$$

$$Z_2^{[1](2)} = W_2^{[1]T}{X^{(2)}}+b_2^{[1]} $$, $$a_2^{[1](2)} = {\sigma}(Z_3^{[1](2)})$$
## Activation functions
The activation function is a non linear function that allows our network to compute complicated problems using only small number of nodes. When working with neural network, one of the choices you get to make is what activation function to use. The activation function is applied elementwise. The different types and the most common are the sigmoid, tanh, ReLU, Leaky ReLU, and maxout. The ReLU is know as the Rectified Linear Unit, it has a very simple shape and it is the commonly used activation function. 
# Fun example: Predicting car prices
With the little background and knowledge you have got so far, let us see how we can apply neural network in prediction the price of a car. We decided to predict the prices of cars since this is a practical example most people can relate with. For this example, we are going to only consider one model of car. The car brand we're considering is Toyota Camry with the following features: The age of the car, the Km travelled, the fuel type and we are going to predict the price of the car. Due to the limited data we have and some important feautures missing, our model will certainly not be perfect since these important missing features also impact the prices of cars. However, the idea is to use example and dataset most people can relate with and also to keep things very simple. 
## Data exploration
Before we start our predictive analyses using neural network, let us first take a visual look of our data and try to understand how each of the features are distrubuted against the price. Here is the first 5 lines of the Toyota camry file that shows us what the data actually look like. 
## Data normalization
One of the most important steps in machine learning is called $${\textbf{normalization}}$$, it is also known as feauture scaling or in simple term data preprocessing. Since our neural network model only works with numbers, the idea of feauture scaling is to convert the data to almost thesame scale and as small as possible. Normalizing the input features can speed up learning by making computation faster.

In our example, the km travelled are between 0 and 350km, the fuel type is binary diesel/petrol, the age is between 0 and 40 and the price ranges between $500 and $40k. 
We are going to normalize the km travelled and the age using mean and variance in order to bring them to thesame scale. Since the fuel type is binary, we're going to transform it to values of -1 and +1. Since what we are going to be predicting is the price and the output of our neural network is going to be between 0 and 1, it will be good for us to normalize this between [0, 1].
For both the km travelled and the age, the normalized equations we will be using are expressed as follow;

$$\varphi = \frac{1}{m}\sum_{i=1}^{m} x_{i}$$ 

$$ X = x_{i} - \varphi $$ 

$$\sigma^{2} = \frac{1}{m}\sum_{i=1}^{m}{(x_{i} - \varphi)}^{2}$$ 

$$ X = \frac{X} {\sigma^{2} }$$

The formular for the normalized car price is expressed as:

$$\frac{x_i - min(x)}{max(x) - min(x)}$$



```ruby
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
```

Also checking out this maths function witk LateX:

$$\sum_{i=1}^{\infty} \frac{1}{n^s} 
= \prod_p \frac{1}{1 - p^{-s}}$$

$$a_{(1)}^{[2]} + a_{(2)}^{[2]} = a_{(3)}^{[2]}$$

$$y = {\sqrt{x^2+(x-1)} \over x-3} + \left| 2x \over x^{0.5x} \right|$$

$$\dfrac{x^2}{x^3} = \dfrac{1}{x}$$


$$\begin{bmatrix}
    x_{11}       & x_{12} & x_{13} & \dots & x_{1n} \\
    x_{21}       & x_{22} & x_{23} & \dots & x_{2n} \\
    \dots 	 &\dots	  &\dots   & \ddots &\dots   \\
    x_{d1}       & x_{d2} & x_{d3} & \dots & x_{dn}
\end{bmatrix}
=
\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{d1} & x_{d2} & x_{d3} & \dots  & x_{dn}
\end{bmatrix}$$



Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
