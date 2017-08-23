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
Now that you have a better understanding of a neural network representation, we are now going to see what these different layers are computing. The first is $$ Z_1^{[1]} = W_1^{[1]T}{x}+b_1^{[1]} $$ and $$a_1^{[1]} = {\sigma}(Z_1^{[1]})$$. For both $$Z$$ and $$a$$, the notation convention is $$a_i^{[l]}$$, where $$[l]$$ denotes the layer number and $$(i)$$ denotes the node in that layer.

For a neural network representation of our example, we will have the following set of equations;

$$ Z_1^{[1]} = W_1^{[1]T}{x}+b_1^{[1]} $$, $$a_1^{[1]} = {\sigma}(Z_1^{[1]})$$

$$ Z_2^{[1]} = W_2^{[1]T}{x}+b_2^{[1]} $$, $$a_2^{[1]} = {\sigma}(Z_3^{[1]})$$

$$ Z_3^{[1]} = W_3^{[1]T}{x}+b_3^{[1]} $$, $$a_3^{[1]} = {\sigma}(Z_3^{[1]})$$







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
