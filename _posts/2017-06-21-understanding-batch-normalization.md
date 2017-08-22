---
title:  "Understanding Batch Normalization"
header:
  teaser: "https://farm5.staticflickr.com/4076/4940499208_b79b77fb0a_z.jpg"
categories: 
  - Neural network
tags:
  - Neural network
  - Machine learning
mathjax: true
---

Understanding batch normalization: 
I’m currently taking Andrew’s [Deep Learning course][deep-learning-course] on Coursera, this course teaches you how to effectively design a neural network from scratch. It is extremely useful in understanding what is happening behind the scene. The bottom-up approach of the course makes it really interesting. The way Andrew breaks down some of these seemly complex techniques and algorithms in the course is joy to watch. From batch normalisation to mini-batch gradient descent to hyperparameters tuning.
That being said, let us get down to the business we have in hand today. Implementing batch normalization. In our previous post where we looked at implementing neural network from scratch. The first thing we did was to pre-processed or normalized our data. We understood that normalizing the input features can speed up learnings. What we did was to compute the mean, the variances and then normalized the data according to the variances.

$$\varphi = \frac{1}{m}\sum_{i=1}^{m} x_{i}$$ 

$$ X = x_{i} - \varphi $$ 

$$\sigma^{2} = \frac{1}{m}\sum_{i=1}^{m}{(x_{i} - \varphi)}^{2}$$ 

$$ X = \frac{X} {\sigma^{2} }$$

![image-center]({{ site.url }}{{ site.baseurl }}/assets/images/NNcontour.png){: .align-center}

As we saw in the post, normalizing the input can turn the contours of our learning problem from a very elongated shape to something very more rounded which makes it easier for our gradient descent algorithm to optimize.

![image-center]({{ site.url }}{{ site.baseurl }}/assets/images/NNbatchnorm.png){: .align-center}

Now for a deeper model where we have not only the input features but several hidden activation layers in our network, and we want to normalize the hidden layers and not just the input features of our model. It would be nice to normalize the mean and variances of the activation function of the previous layer. This will make the training of our paramters more efficient. 
For any hidden layer, can we normalize the value of say $$ Z^{[2]}$$ so as to train the values of $$ W^{[3]}, b^{[3]} $$ faster and make our model more efficient.



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

$$W^{1} = \begin{bmatrix}  
0.01 & 0.05 & 0.07 \\  
0.20 & 0.041 & 0.11 \\  
0.04 & 0.56 & 0.13  
\end{bmatrix}$$


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
[deep-learning-course]: https://www.coursera.org/learn/neural-networks-deep-learning
