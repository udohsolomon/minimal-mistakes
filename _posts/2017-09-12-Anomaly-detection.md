---
title:  "Anomaly detection algorithm implemented in Python"
header:
  teaser: "https://farm5.staticflickr.com/4076/4940499208_b79b77fb0a_z.jpg"
categories: 
  - Neural network
tags:
  - Neural network
  - Machine learning
mathjax: true
---

This post is an overview of a simple anomaly detection algorithm implemented in Python. While there are different types of anomaly detection algorithms, we will focus on the univariate and the multivariate gaussian distribution algorithms in this post. 

Anomaly detection problem is the identification of outliers in data points relative to some standard or expected outcome. It has significant wide range of applications in several industries due to the critical and actionable information it provides. Some of the examples of anomaly detection are fraud detection in an online transaction, an unexpected growth of users on a website in a short period of time that looks like a spike, another is a production plant where the sensors may indicate faulty components and monitoring of computer servers in a data centre. 

The basic approach of anomaly detection is defining a boundry around the normal data points that separates them from the outliers. However, several factors make this approach very challenging and it's not very straightforward in itself. 

In this post, we will implement anomaly detection algorithm to detect outliers in computer servers in a data centre for monitoring purpose. The Gaussian distribution model is used for this example. 
