---
layout: post
title: "Projects"
author: "Max"
permalink: /projects/
---
favicon
update home page
analytics

## AI Poker
(This section is repeated on the [Poker](https://chisness.github.io/poker/) page)

### [AI Poker Tutorial](https://www.aipokertutorial.com)
Over the years, poker strategy has become more and more mathematical and many top players play by studying game theory optimal (GTO) strategy with "solver" software. This tutorial touches on how the math and algorithms behind the solvers work as well as important poker research papers and ideas. 
- In progress

### [Most Important Fundamental Rule of Poker](https://arxiv.org/abs/1906.09895)
Using techniques from machine learning we have uncovered a new simple, fundamental rule of poker strategy that leads to a significant improvement in performance over the best prior rule and can also easily be applied by human players.
- 2020 with [Sam Ganzfried](http://www.ganzfriedresearch.com/)

### [Counterfactual Regret Minimization and Abstraction Analysis in Computer Poker](https://www.dropbox.com/s/jcgszjng6u5gj0b/MaxChiswickCFRThesis.pdf?dl=0)
MSc thesis analyzing card abstraction in Kuhn (1-card) poker and betting abstraction in No Limit Royal Hold'em (same as No Limit Texas Hold'em, but with a 20-card deck with cards Ten and higher only).
- 2017 Technion Israel Institute of Technology with Professor Nahum Shimkin

## Gambling Applications
### Jan 2020: [Policy and Value Iteration Gambler Problem](https://chisness.github.io/2020-01-14/policy-and-value-iteration-gambler-problem)
- Analyzing the gambler problem from the Sutton Barto reinforcement learning textbook

### Jul 2020: [Freethrow Bet Evaluation](https://chisness.github.io/2020-07-10/freethrow-bet-evaluation)
- Reinforcement learning to optimize the strategy for [Mike McDonald](https://twitter.com/MikeMcDonald89)'s free throw bet where he had to make 90/100 free throws and was allowed to reset his attempts at any point
- [Freethrow Bet Evaluation TLDR Version](https://chisness.github.io/2020-08-20/freethrow-bet-tldr)
- [Freethrow Bet Evaluation on Medium](https://towardsdatascience.com/apply-reinforcement-learning-to-win-a-free-throw-bet-f555b8adc0de)

### Sep 2020: [Monte Carlo RL and Blackjack](https://chisness.github.io/2020-01-21/monte-carlo-rl-and-blackjack)
- Finding optimal blackjack strategies using Monte Carlo reinforcement learning

## AI Safety
### Sep 2020: Assessing Generalization in Reward Learning
- [Part 1: Intro and Background](https://medium.com/@chisness/assessing-generalization-in-reward-learning-intro-and-background-da6c99d9e48)
- [Part 2: Implementations and Experiments](https://towardsdatascience.com/assessing-generalization-in-reward-learning-implementations-and-experiments-de02e1d08c0e)
- One of the primary goals in machine learning is to create algorithms and architectures that demonstrate good generalization ability to samples outside of the training set. In reinforcement learning, however, the same environments are often used for both training and testing, which may lead to significant overfitting. We build on previous work in reward learning and model generalization to evaluate reward learning on random, procedurally generated environments. We implement algorithms such as T-REX (Brown et al 2019) and apply them to procedurally generated environments from the Procgen benchmark (Cobbe et al 2019). Given this diverse set of environments, our experiments involve training reward models on a set number of levels and then evaluating them, as well as policies trained on them, on separate sets of test levels.
- Work as part of the 2020 AI Safety Camp with Anton Makiievskyi and Liang Zhou

## Reinforcement Learning
### Soon

## Machine Learning/Deep Learning
### Apr 2020: [Prediction of Bayesian Intervals for Tropical Storms](https://arxiv.org/abs/2003.05024)
- Building on recent research for prediction of hurricane trajectories using recurrent neural networks (RNNs), we have developed improved methods and generalized the approach to predict a confidence interval region of the trajectory utilizing Bayesian methods. Tropical storms are capable of causing severe damage, so accurately predicting their trajectories can bring significant benefits to cities and lives, especially as they grow more intense due to climate change effects. By implementing the Bayesian confidence interval using dropout in an RNN, we improve the actionability of the predictions, for example by estimating the areas to evacuate in the landfall region. We used an RNN to predict the trajectory of the storms at 6-hour intervals. We used latitude, longitude, windspeed, and pressure features from a Statistical Hurricane Intensity Prediction Scheme (SHIPS) dataset of about 500 tropical storms in the Atlantic Ocean. Our results show how neural network dropout values affect our predictions and Bayesian intervals.
- Accepted to [ICLR 2020 Climate Change Workshop](https://www.climatechange.ai/papers/iclr2020/14.html) and FLAIRS 2020 
- With [Sam Ganzfried](http://www.ganzfriedresearch.com/)

## Python
### Soon

## GitHub
### My GitHub is here: [https://github.com/chisness](https://github.com/chisness)
