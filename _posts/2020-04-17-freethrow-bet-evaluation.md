## The Bet
[Mike McDonald](https://twitter.com/MikeMcDonald89) is a successful gambler/poker player who set up a bet with the [following terms](https://twitter.com/MikeMcDonald89/status/1246917870677680129):

Main terms: 
- I must sink 90/100 free throws on an attempt. I get unlimited attempts.
- Regulation ball, regulation hoop, lane violations not allowed
- Even money
- Bet lasts until end of 2020

Detailed additions: 
- I must define when an attempt starts. I.e. if I shot 150 and scored 90/100 between #20-119 I'd need to reset counter after 19 for it to count.
- If I have no safe access to a regulation hoop I get an extension until I have had 50 hours of safe use of a regulation hoop

The short version is that Mike has to get 90/100 freethrows by the end of 2020 with unlimited attempts and unlimited time per attempt, but each attempt has to be declared. 

## Assumptions
It seems that the "hot hand" theory of improving chances of making a basket on a "hot streak" is pretty unclear ([Wikipedia article](https://en.wikipedia.org/wiki/Hot_hand)) and also would make the analysis much more complicated, so we assume a fixed probability of making each shot. 

There are two main elements that go into the bet: Skill level and reset strategy. The skill level is defined as the proability of making each shot. The reset strategy is when to start a fresh 100 shot attempt. We suggest that if the probability of success from any point is worse than the probability of success from the starting point, then it's best to reset. 

[NOTE: shouldn't we have some threshold here to account for time, so prob. success if very close to end is better than same prob. success at beginning?]

## Probability of making 90/100


## When to reset attempts? Method 1: Binomial




## When to reset attempts? Method 2: Reinforcement Learning
We can analyze the reset attempt problem using reinforcement learning value iteration. Here's how that works. 

Initial definitions: 
- We define a state as a pair [shots made, shots missed]
- You can think of this as a table with shots missed on the x-axis and shots made on the y-axis
- Shots missed goes from 0 to 10
- Shots made goes from 0 to 90
- We define each pair to have a reward of 0 except for every combination where shots made = 90, so [90, 3] and [90, 5] and [90, 10], etc. all have a reward value of 100 (chosen arbitrarily)
- We define 2 possible actions at each state: shoot or reset. These represent the actions of the player in the bet. 

Value iteration: 
- We cycle through every combination of [shots made, shots missed] 


[NOTE: do policy evaluation?]


## Binomial vs. RL and Conclusions