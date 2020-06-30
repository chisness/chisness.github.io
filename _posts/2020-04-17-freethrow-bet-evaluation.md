This post was written jointly by Max Chiswick and [Mike Thompson](https://www.linkedin.com/in/mike-thompson-78655b13/)

## The Bet
[Mike McDonald](https://twitter.com/MikeMcDonald89) is a successful gambler/poker player who set up a bet with the [following terms](https://twitter.com/MikeMcDonald89/status/1246917870677680129):

Main terms: 
- I must sink 90/100 free throws on an attempt. I get unlimited attempts.
- Regulation ball, regulation hoop, lane violations not allowed
- Even money
- Bet lasts until end of 2020

Detailed additions: 
- I must define when an attempt starts. I.e. if I shot 150 times and scored 90/100 between attempts 20-119 I'd need to reset counter after 19 for it to count.
- If I have no safe access to a regulation hoop I get an extension until I have had 50 hours of safe use of a regulation hoop

The short version is that Mike has to get 90/100 freethrows by the end of 2020 with unlimited attempts and unlimited time per attempt, but each attempt has to be declared. 

## Assumptions and Simplifications
It seems that the "hot hand" theory of improving chances of making a basket on a "hot streak" is pretty unclear ([Wikipedia article](https://en.wikipedia.org/wiki/Hot_hand)) and also would make the analysis much more complicated, so we assume a fixed probability of making each shot. 

There are two main elements that go into the bet: Skill level and reset strategy. The skill level is defined as the proability of making each shot. The reset strategy is when to start a fresh 100 shot attempt. We suggest that if the probability of success from any point is worse than the probability of success from the starting point, then it's best to reset. (Although in reality, it makes sense to prefer to continue when the probability is the same or slightly worse than the starting probability since it will take more time to start over.)

## Probability of making 90/100
If we assume a fixed probability of making each shot equal to $$p$$, then the probability of making at least 90 out of 100 shots is a binomially distributed random variable, with probability of success equal to: $$\sum_{i=90}^{100} {100 \choose i} * p^i*q^{100-i}$$; where $$q$$ is the probability of a miss = $$1 - p$$.

We can evaluate the probability of success by plugging in various values for true shooting percentage ($$p$$ in the equation above). 
The graph below demonstrates the impact that true shooting percentage has on probablity of success:
![Single Attempt](../assets/attempt1.png)

We can see that a shooter of around 70% or below is quite unlikely to ever hit 90 out of 100. Using $$p$$ = 70% results in a probability of 1 in 642,853, which clearly is an unrealistic number of attempts in a one year period.

How, then, can we determine a true shooting percentage that is likely to be successful? If $$x$$ equals the probability of success on one attempt (equal to the equation above), then $$(1-x)$$ is the probablity of failure on one attempt and $$(1-x)^n$$ is the probability of failure on all n attempts, then $$1-(1-x)^n$$ is the probability of at least one success over $$n$$ attempts. 

For simplicity, let's assume one attempt per day and again examine the probability of success for various true shooting percentages:
![365 Attempts](../assets/attempt365.png)
We can see that somewhere between 78% and 79% has a 50% probability of achieving success if they make 1 attempt per day. Anyone with a true shooting percentage in the low 70's would have a marginal probability of success, and below 70% it is quite unlikely to ever make 90 out of 100 free throws. Anyone that shoots 80% or above is almost guaranteed to be successful after 365 attempts. 

## When to reset attempts? Method 1: Binomial
Since anyone outside of high 70's is either almost guaranteed to fail (if below) or succeed (if above) then the question of whether he will be successful really is only interesting if his true shooting percentage falls somewhere in that range. Let's assume a true shooting percentage of 78%, which has a 40% probability of success over 365 attempts. At what point should Mike reset his attempt back to 0 if he has missed a few shots? Obviously if he misses the first shot he should reset, and probably even if he misses the second or third shot, but what about the seventh shot? What if he is 35 of 40? 

One strategy is to compute the probability of success at that point in time, and reset if it is lower than his probability of success at the start of an attempt. A 78% true shooter has a 0.14% (or 1 in 709) chance of making at least 90 out of 100 free throws. If after taking $$x$$ shots, he has $$y$$ misses, his probability of success is: $$\sum_{i=90-x+y}^{100-x} {100-x \choose i} * 0.78^i*0.22^{100-i-x}$$

We can find the decision boundary by finding the maximum number of attempts for given number of misses $$y$$ = 1 through 10 such that the probability of success is lower at that state than the probability of success at the beginning of an attempt:
![reset](../assets/reset_graph.png)

The datapoint 8 misses, 55 total attempts means that if his 8th miss was after 55 or fewer total shots, he should reset the attempt because his probability of success is lower than at the beginning of an attempt.

### Issues with Binomial Analysis
In reality, a person does not have a fixed true shooting percentage. The probability of making a shot will vary day by day and even throughout an attempt. A more accurate model would treat true shooting percentage as a random variable rather than a fixed quantity. Since success is a tail event (in the situation where the true shooting percentage is in the high 70's), added variance around true shooting percentage will make the probability of success more likely. Additionally, there is debate around whether there is autocorrelation between attempts, which would also add variance.

Also, the strategy above does not take into account time. He likely would accept a slightly lower probability of success at states that already have a high number of shots taken. This would increase the total number of shots threshold for a given miss value.

## When to reset attempts? Method 2: Reinforcement Learning
We can analyze the reset attempt problem using reinforcement learning value iteration. Here's how that works. 

Initial definitions: 
- We define a state as a pair [shots made, shots missed]
- You can think of this as a table with shots missed on the x-axis and shots made on the y-axis
- Shots missed goes from 0 to 10
- Shots made goes from 0 to 90
- We define each pair to have a reward of arriving to that state of 0 except for every combination where shots made = 90, so [90, 3] and [90, 5] and [90, 10], etc. all have a reward value of 100 (chosen arbitrarily to represent winning)
- We define 2 possible actions at each state: shoot or reset. These represent the actions of the player in the bet. 

We want to use value iteration to find the value of every state [shots made, shots missed]. Note that the immediate reward of 100 is only given for winning the bet, so now we are defining state values that derive from that winning reward. The state values say what it's worth to be in any given state given that winning has a reward of 100 and given a defined discount rate, $$\gamma$$.

We have defined the reward for winning (i.e. 100 when shots made reaches 90) and can initially set the value of every other state at 0. Then the algorithm will learn the value of those positions. For example, if you are in the state of [89 made, 10 missed], your value is 100 if you make the next shot and you will be back to the beginning if you miss it. So we can say that the value of that state is = $$p_make*100 + (1-p_make)*\text{[value of starting state]}$$. 

Here's how value iteration works: 
<script src="https://gist.github.com/chisness/0a7778093ff0b77f5fd01215b32f26e5.js"></script>
- We cycle through every combination of [shots made, shots missed] 
- We set a variable $$v$$ to the current value of each state
- For each state, we evaluate the value of (1) shooting and (2) resetting
- Reset is the current value of the state [0, 0]
- Shooting has a probabilistic result and we use the Bellman equation to evaluate, which is [BELLMAN EQUATION]
- In the case of making, we have: $$make_val = p_make*$$
- In the case of missing, we have:  $$miss_val = p_miss*$$
- After these calculations, we have a result for the reset action and the shoot action
- We can then set the value of the state as the result for which action gives us the greatest value
- After each state that we check, we compare the original value of the state that we stored as $$v$$ to the new value. We keep track of the largest difference as we go through each state so that at the end of the cycle, we can see what the largest difference is. Once this difference converges below some $$\epsilon$$ value that we define, then we consider the state values to be stable. 
<script src="https://gist.github.com/chisness/762272f794d0fb8eadd683778c9ed30a.js"></script>
- Now we have values for each state, but we haven't made a strategy (aka policy) for what to do at each state. We can iterate through every state pair one final time and check the value of each action at each state and now that these values are fixed, we can set the strategy for the state to be the action that gives the highest value. This is called policy iteration. 

## The discount rate
We use the parameter $$\gamma$$ in the Bellman equation. This acts as a discount rate, which means that farther away states get discounted more compared to states nearby. We think this makes sense in the context of the freethrow bet because of the time and energy required to complete attempts. For example, if we had a perfect player who could make every shot 100% of the time, if he had 1 shot left, the value of the state would be $$100 * 0.99 = 99$$ and with 5 shots left would be $$100 * 0.99^5 = 95.099 and then at the beginning would be $$100 * 0.99^90 = 40.473$$. 

Lowering threshold for continuing by gamma value 

p_make = 0.78, $$\gamma$$ = 1

p_make = 0.78, $$\gamma$$ = 0.995 #maybe .999

p_make = 0.78, $$\gamma$$ = 0.99

p_make = 0.78, $$\gamma$$ = 0.9

pick 1 gamma
do p_make = 0.5, 0.7, 0.78, 0.95

show 1 plot with everything and rest 

do sims with naive way no resets, binomial, and a couple of gamma levels 

how to show reset strategy is valuable (avg length of attempt with reset compared to without?)

1st miss 7th shot or later continue

post on 2+2 when done

what about cost per shot 

## Binomial vs. RL and Conclusions
Having the discount rate built into the reinforcement learning model is a solution for the issue of considering the value of time. 

## Practical Strategy
We see that the reset strategies are all fairly similar and all have used the simplifying assumption of a fixed freethrow shooting make percentage. If I were playing (and thank god I'm not with my likely make percentage), I would look at the range of reset numbers and always reset below, never reset above, and then evaluate based on my perceived streakiness if in between