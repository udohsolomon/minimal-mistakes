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

![image-center]({{ site.url }}{{ site.baseurl }}/assets/images/NNnetwork.png){: .align-center}

Now for a deeper model where we have not only the input features but several hidden activation layers in our network, and we want to normalize the hidden layers and not just the input features of our model. It would be nice to normalize the mean and variances of the activation function of the previous layer. This will make the training of our paramters more efficient. 

For any hidden layer, we can normalize the value of say $$ Z^{[3]} $$ in hidden layer $$ L^{4} $$ so as to train the values of $$ W^{[4]}, b^{[4]} $$ faster and make our model more efficient.

Considering our deep learning example from the perspective of a certain layer, say the 4th hidden layer, $$ L^{4} $$. So our network has to learn the parameters $$ W^{[4]}, b^{[4]} $$. From the perspective of this 4th hidden layer, each of the preceeding layers have great influence over what the input of this layer will see. As you start to train your network, the distribution of what this layer sees will vary significantly with time. As an analogy, let us say you train your dataset on all images of black cats, if you try to apply this network to dataset with coloured cats where the positive examples are not just black cats, then your classifier or prediction will perform poorly.

This idea of your data distribution changing is known as $$\textbf {Covariate shift} $$. The idea is that if you've learned some $$X $$ to $$Y$$ mapping $$ X \rightarrow Y $$, and at any time the distribution of $$ X $$ changes, then you might need to retrain your learning algorithm.

So what batch norm does is to reduce the the amount that the distribution of the hidden units shift around. What batch norm does is that no matter how the parameter of the previous layer changes, their mean and variance will remain thesame. It limits the amount to which updating the parameters of the earlier layers can can affect the distribution of values that the current layer now sees and therefore has to learn on. This makes the values of the current layer to become more stable and provide the later layers more firm ground to stand on. Another interesting thing about batch norm is that it has a slight regularization effect. Though this is really not the intent of batch norm but sometimes it has this intended or unintended effect on your learning algorithm. 
## Implementing Batch Normalization
Given some intermediate values in a neural network, we can add batch norm by first, feeding the input $$X$$ into the first hidden layer $$ L^{1} $$ and then compute $$ Z^{[1]} $$ govern by the parameters $$ W^{[1]}, b^{[1]} $$. We then take this parameter $$ Z^{[1]} $$ and apply batch norm govern by the parameters $$ {\beta^{[1]}}$$ and $${\gamma^{[1]}}$$. This will give us the new normalized value $${\hat{Z}^{[1]}}$$, which we then feed into the activation function to give us $${a}^{[1]} = {g}^{[1]}{({\hat{Z}^{[1]}})}$$.

$$ {X} \xrightarrow{W^{[1]}, b^{[1]}} {Z^{[1]}} \xrightarrow[Batch Norm (BN)]{\beta^{[1]}, \gamma^{[1]}} {\hat{Z}^{[1]}}\rightarrow{a}^{[1]} = {g}^{[1]}{({\hat{Z}^{[1]}})} $$

Now we've done the computation for the first hidden layer $$ L^{1} $$. Next, we take the value $${a}^{[1]}$$ and use it to compute the batch norm for the next hidden layer and so on. 

$${a}^{[1]} \xrightarrow{W^{[2]}, b^{[2]}} {Z^{[2]}} \xrightarrow[BN]{\beta^{[2]}, \gamma^{[2]}} {\hat{Z}^{[2]}}\rightarrow{a}^{[2]}$$

With these new set of parameters in our algorithm, we can then use whatever optimization technique we want to. So far, we've been talking about batch norm as if we were training on the entire training set at thesame time, like we're using batch gradient descent. However, It is worth noting that in practice, batch normalization is usually applied with mini-batches in training set. 



$$\begin{verbatim}
for t = 1 ... n(mini-batches)
  compute forward propagation on $$X^{t}$$
	In each hidden layer, use BN to replace $$Z^{[l]}$$ with $${\hat{Z}^{[l]}}$$
  compute  $$dW^{[l]}$$, $$d{\beta}^{[l]}$$, $$d{\gamma}^{[l]}$$ using backward propagation
  
  update parameters $${dW}^{[l]} := {dW}^{[l]} - {\alpha}{dW}^{[l]} $$
		    $${\beta}^{[l]} := {\beta}^{[l]} - {\alpha}d{\beta}^{[l]} $$
		    $${\gamma}^{[l]} := {\gamma}^{[l]} - {\alpha}d{\gamma}^{[l]}
\end{verbatim}$$

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
