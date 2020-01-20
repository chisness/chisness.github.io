## The Gambler Problem
We are going to illustrate value iteration and policy iteration with the Gambler Problem from the Reinforcement Learning book by Sutton and Barto. 

The gambling event is to bet on coin outcomes such that heads is a winning wager and tails is losing. The probability of heads is selected as a variable and the gambler's goal is to get to a score of 100. Getting to 0 is a loss. The gambler can have betting states from 1, 2, ..., 99. 

Bets can be made up to the amount of money that the gambler has, but not an amount that would bring the total to over 100 in the event of a winning bet. 

For example, if the gambler has 60, the maximum bet is 40 because anything over 40 would result in a score of over 100 with a win. 

So we have betting options of 0, 1, ..., min(s, 100-s), where the state is from 1 to 99. 

We will use value iteration and policy iteration to find the optimal policy and game value at each state. 

## Value Iteration
We initialize a state value V(s) and policy pi(s) for each state as zeros. Then we iterate through each state from 1 to 99. In each iteration, we set v as the value of that state, determine the range of possible actions, and set val_action as zeros for each possible action. 

Then we loop through each possible action and evaluate its value using the Bellman equation. 

The Bellman equation effectively calculates the value of an action at a state by computing the expected value of the immediate return from the action and (in the case of 1-step lookahead) the currently known value of the resulting state. 

We set the value of the state as the maximum of all of the action values. 

This value iteration over every state keeps running until the maximum change of the state value over all states is less than a preset epsilon. To do this, we initialize a delta value as 0 and update the delta after each state iteration to be the maximum of the previous delta or the absolute value of the change in that state value. 

Once we finish an iteration of all state values and delta is less than the epsilon value, we assume that we have found the state values and then find the optimal policy given these state values. 

This is a single for-loop over every state from 1 to 99 in which we calculate the action value with the Bellman equation for each action for each state and then set the best action as the action that results in the highest value. We set this as the policy for that state. 

## Policy Iteration




## Code