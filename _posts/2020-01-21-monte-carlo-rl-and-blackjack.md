---
layout: post
title: "Monte Carlo RL and Blackjack"
---

**Contents**
* TOC
{:toc}

## Monte Carlo RL and Blackjack
The Reinforcement Learning book by Sutton and Barto has a blackjack example for prediction and control with Monte Carlo reinforcement learning. Basic rules for blackjack are viewable on [Wikipedia](https://en.wikipedia.org/wiki/Blackjack). The short version of the rules is that the player and the dealer are each dealt 2 cards to start. Cards are worth their stated (face) values, with all face cards worth 10, and Aces can be used as 1 or 11. 21 is the best possible hand and if you go over 21, then you automatically lose (aka bust). Aces effectively count as 11 unless they would make you bust, in which case you can count them as 1. 

The goal is to beat the dealer, who has the advantages of going last (so even if you both bust, the player loses when they bust first). The player gets to see one dealer card, which majorly influences the decisions. The dealer plays with fixed rules that say to hit on 16 and below and stand on 17 and above (this is fairly standard and is what we use but may vary slightly from casino to casino). The player can hit or stand whenever they would like. 

Link to full code: 

## The OpenAI Gym Environment and Modifications
There is a built-in [OpenAI Gym blackjack environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py) available to use in the gym's toy_text directory. This environment is quite basic and handles the most standard rules, which include getting paid 1.5x your bet if you get a natural blackjack (i.e. blackjack on the first 2 cards) and the dealer hitting until their hand is >= 17. The environment uses cards from an "infinite" deck to simplify (most casinos use 6-8 decks so players can't as easily count cards and know when the deck will be favorable).

The environment allows players to hit and stand, which gets the point across, but almost all blackjack games allow for doubling down and splitting. Doubling down means that you can double your bet after seeing your 2 initial cards, but only get 1 total extra card. Splitting means that you can split a pair and turn it into two hands, each with the same bet as the original (so the overall bet is doubled). 

I implemented doubling down into the Blackjack environment to make it a little more realistic. I copied the original blackjack.py gym file, made the updates to add in the double down state, and then saved as blackjack1.py. I put this file into the same directory as the code file and added the following lines of code: 

'''python
register(id='BlackjackMax-v0', entry_point='blackjack1:BlackjackEnv1')
ENV_NAME = "BlackjackMax-v0"
'''

blackjack1.py is available here: 

We will use the standard version except in one section where we explore the differences when allowing for doubling down. 

## Monte Carlo RL 
Monte Carlo RL in blackjack works by running out single hand samples repeatedly and deriving state values from the averages of the returns of each of these runouts. This will be explained more in the following sections. A state in this game is defined as:

[our cards, dealer card showing, whether our Ace is "playable"]
 
A playable Ace means that the Ace is being used as 11 instead of 1. An ace being used as 11 is also referred to as a "soft" hand and when used as 1 is a "hard" hand -- soft because when you have, for example, an Ace and a 6 (soft 17), you're able to take another card without any risk of busting, because you can just use the Ace as a 1 if your card is above a 4. If you got a 4, you would use the Ace as 11 and have the 6 and 4 for 21. If you got an 8, you'd use the Ace as a 1 and have the 6 and 8 for a hard 15, and then would have the option to hit again. 

### Implementation
We simulate hands and append each state and action from the hand to an "episode" list. Episode is the generic term for a sequence of states and actions. 

When the hand finishes, each state and action pair that was seen has a "total return" sum that is incremented by the final reward (i.e. winnings/losings from the result of the hand) and a "number of times seen" counter for that pair is incremented. These track the total rewards earned from each state and 

'''python
for i in range(1000000):
	while True:
		action_probs = agent.play_action(new_state)
		action = np.random.choice(BETS, p=action_probs)
		episode.append((new_state, action))
		new_state, reward, done, _ = agent.env.step(action)
		if done:
			for (state, action) in episode:
				returns_sum[(state,action)] += reward
				returns_count[(state,action)] += 1
				agent.values[(state, action)] = returns_sum[(state,action)] / returns_count[(state,action)]
'''

After updating the values, we also need to update the policy. We first determine the best action given our value table for each state. Then we use a "soft greedy" policy rule that assigns the best action probability $$1 - \epsilon + \frac{\epsilon}{3}$$, while assigning the other 2 actions $$\frac{\epsilon}{3}$$ so the other actions both have some low probability of being chosen. 

'''python
for (state, _) in episode:
	vals = [agent.values[(state, a)] for a in BETS]
	best_action = np.argmax(vals)
	for a in BETS:
		if a == best_action:
			agent.policy[state][a] = 1 - EPSILON + EPSILON/len(BETS)
		else:
			agent.policy[state][a] = EPSILON/len(BETS)
'''

## Monte Carlo Prediction (Evaluating a fixed policy)
We can use Monte Carlo Prediction to evaluate a fixed blackjack strategy. Let's consider a strategy (as given as an example in the Sutton Barto RL book) that sticks on 20 and 21 and hits on everything else. Note that this strategy is actually quite bad because it (a) doesn't incorporate knowledge of the dealer's hand and (b) hitting on 17/18/19 is generally not a good idea! 

After a large number of simulations, we get approximate state-value functions for 


## Monte Carlo Control (Approximating optimal policies)

We begin with the player using a uniform random strategy, so if the options are stick, hit, and double, each initially takes a value of 1/3. 


Then we determine the best action for each state that was seen in the episode and use a soft greedy policy such that the current best action is played $$1 - \epsilon + \frac{epsilon}{3}$$ and the other two actions are played with probabilities $$\frac{\epsilon}{3}$$. The soft greedy policy is used so that we still explore alternative actions even if one action does very well at the beginning. 

After a large number of simulations, the state values and policy are set as above. We can then evaluate the policy by running simulations and using the fully greedy policy (since exploration is no longer needed) and computing the total reward over all of the simulations divided by the number of simulations to find the average winnings per hand. 

## Monte Carlo Control with Doubling Down
https://wizardofodds.com/games/blackjack/strategy/4-decks/

## Monte Carlo Control with Doubling Down and Seeing Both Dealer Cards
You're only supposed to see 1 dealer card. What if somehow the other flipped over and you saw both? How big of an advantage would this be? Normal strategy is to never hit on 19 because the risk of busting is so high. But if you saw that the dealer had 20, it would be mandatory to hit! 

## Importance Sampling and Off-Policy

### Importance Sampling Prediction

### Importance Sampling Control