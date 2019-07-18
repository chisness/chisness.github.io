## Intro
This tutorial is made with two target audiences in mind: (1) Those with an interest in poker who want to understand how AI poker agents are made and (2) Those with an interest in coding who want to better understand poker and code poker agents from scratch. 

## Table of Contents
[1. Poker Background](#1-poker-background)<br>
[2. History of Solving Poker](#2-history-of-solving-poker)<br>
[3. Game Theory -- Equilibrium and Regret](#3-game-theory----equilibrium-and-regret)<br>
[4. Solving Toy Poker Games from Normal Form to Extensive Form](#4-solving-toy-poker-games-from-normal-form-to-extensive-form)<br>
[5. Trees in Games](#5-trees-in-games)<br>
[6. Toy Games and Python Implementation](#6-toy-games-and-python-implementation)<br>
[7. Counterfactual Regret Minimization (CFR)](#7-counterfactual-regret-minimization-cfr)<br>
[8. Game Abstraction](#8-game-abstraction)<br>
[9. Agent Evaluation](#9-agent-evaluation)<br>
[10. CFR Advances](#10-cfr-advances)<br>
[11. Deep Learning and Poker](#11-deep-learning-and-poker)<br>
[12. AI vs. Humans -- What Can Humans Learn?](#12-ai-vs-humans----what-can-humans-learn)<br>
[13. Multiplayer Games](#13-multiplayer-games)<br>
[14. Opponent Exploitation](#14-opponent-exploitation)<br>
[15. Other Games (Chess, Atari, Go, Starcraft, Dota, Hanabi)](#15-other-games-chess-atari-go-starcraft-hanabi)<br>

## 1. Poker Background

### Brief History
Poker began at some point in the early 20th century and grew extensively in the early 2000s thanks to the beginnings of online poker (the first hand ever was dealt on January 1, 1998), which lead to an accountant and amateur poker player named Chris Moneymaker investing $39 in an online satellite tournament that won him a $10,000 seat at the World Series of Poker Main Event in 2003, alongside 838 other entrants. Moneymaker went on to defeat professional poker player Sam Farha at the final table of the tournament and won $2.5 million. A poker boom was sparked and led to massive player pools on the Internet and in subsequent World Series’ of Poker.

![Chris Moneymaker](https://pnimg.net/w/articles/4/5a5/a09111f94a.jpg)
https://www.pokernews.com/strategy/teaching-poker-to-beginners-with-chris-moneymaker-29704.htm

After the human poker boom, computers also started getting in on the poker action. Researchers began to study solving Texas Hold’em games in 2003, and since 2006, there has been an Annual Computer Poker Competition (ACPC) at the AAAI Conference on Artificial Intelligence in which poker agents compete against each other in a variety of poker formats. In early 2017, for the first time, a NLHE poker agent defeated and is considered superior to top poker players in the world in 1v1 games.

### Basic Rules
Poker is a card game that, in its standard forms, uses a deck of 52 cards composed of four suits (Clubs :clubs:, Diamonds :diamonds:, Hearts :heart:, and Spades ::spades::) and 13 ranks (Two through Ten, Jack, Queen, King, and Ace). A dealer button rotates around the table indicating who is the “dealer”. This is decided at random for the first hand and rotates clockwise after that. All actions begin to the left of the hand’s current dealer player.

We will focus on two player games and ignore any fees (also known as rake) such that the games will be zero-sum (all money won by one player is lost by the other). The two players play a match of independent games, also called hands, while alternating who is the dealer.

Each game goes through a series of betting rounds that result in either one player folding and the other winning the pot by default or both players going to “showdown”, in which the best hand wins the pot. The pot accumulates all bets throughout the hand. The goal is to win as many chips from the other player as possible.

Betting options available throughout each round are: fold, check, call, bet, and raise. Bets and raises generally represent strong hands. A bluff is a bet or raise with a weak hand and a semi-bluff is a bet or raise made with a currently non-strong hand that has good potential to improve. 

**Fold:** Not putting in any chips and "quitting" the hand by throwing the cards away and declining to match the opponent's bet or raise. Done only after an opponent bet or raise. <br>
**Check:** A pass, only possible if acting first or if all previous players have also checked<br>
**Call:** Matching the exact amount of a previous bet or raise<br>
**Bet:** Wagering money (putting chips into the pot) when either first to act or the opponent has checked, which the opponent then has to call or raise to stay in the pot<br>
**Raise:** Only possible after a bet, adding more money to the pot, which must be at least the amount of the previous bet or raise (effectively calling and betting together)

### Kuhn (1-card) Poker
Kuhn Poker is the most basic useful poker game that is used in computer poker research. It was solved analytically by hand by Kuhn in 1950 [6]. Each player is dealt one card privately and begins with two chips. In the standard form, the deck consists of only three cards – an Ace, a King, and a Queen, but can be modified to contain any number such that the cards are simply labeled 1 through n, with a deck of size n. Our first experiment will use a deck size of 100 to compare different CFR algorithm implementations.

Players each ante 1 chip (although most standard poker games use blinds, this basic game does not) and rotate acting first, and the highest card is the best hand. With only 1 chip remaining for each player, the betting is quite simple. The first to act has the option to bet or check. If he bets, the opponent can either call or fold. If the opponent folds, the bettor wins one chip. If the opponent calls, the player with the higher card (best hand) wins two chips.

If the first to act player checks, then the second player can either check or bet. If he checks, the player with the best hand wins one chip. If he bets, then the first player can either fold and player two will win one chip, or he can call, and the player with the best hand will win two chips.

#### All Kuhn Poker Sequences


### Leduc Poker
Leduc Poker is a simple toy game that has more in common strategically with regular Texas Hold'em. 

**Deck size:** 6 cards -- 2 Jacks, 2 Queens, 2 Kings<br>
**Rounds:** 2 rounds -- 1 private card preflop, 1 community flop card<br>
**Blinds/Antes:** $1 ante<br>
**Betting structure:** All bets $2 in the first round and $4 in the second round with a limit of one bet and one raise per round<br>
**Starting 

Each betting round follows the same format. The first player to act has the option to check or bet. When betting the player adds chips into the pot and action moves to the other player. When a player faces a bet, they have the option to fold, call or raise. When folding, a player forfeits the hand and all the money in the pot is awarded to the opposing player. When calling, a player places enough chips into the pot to match the bet faced and the betting round is concluded. When raising, the player must put more chips into the pot than the current bet faced and action moves to the opposing player. If the first player checks initially, the second player may check to conclude the betting round or bet. In Leduc Hold’em there is a limit of one bet and one raise per round. The bets and raises are of a fixed size. This size is two chips in the first betting round and four chips in the second.

#### Example Leduc Poker Hand

### Texas Hold'em Poker
Each hand starts with the dealer player posting the small blind and the non-dealer player posting the big blind. The blinds define the stakes of the game (for example, a $1-$2 stakes game has blinds of $1 and $2) and the big blind is generally double the small blind. They are called blinds because they are forced bets that must be posted “blindly”. The player to the left of the big blind, in this case the dealer player, begins the first betting round by folding, calling, or raising. (In some games antes are used instead of or in addition to blinds, which involves each player posting the same ante amount in the pot before the hand.)

Each hand in Texas Hold’em consists of four betting rounds. Betting rounds start with each player receiving two private cards, called the “preflop” betting round, then can continue with the “flop” of three community cards followed by a betting round, the “turn” of one community card followed by a betting round, and a final betting round after the fifth and final community card, called the “river”. Community cards are shared and are dealt face up.

In no limit betting, the minimum bet size is the smaller of the big blind or a bet faced by the player and the maximum bet size is the amount of chips in front of the player. In the case of a two-player game, the dealer button pays the small blind and acts first preflop and then last postflop.
In limit betting, bets are fixed in advance based on the stakes of the game and the blinds. For example, with 2-4 blinds, the bets preflop and on the flop are 4 and on the turn and river, they are doubled to 8. In limit betting, there is a maximum of four bets and raises per betting round per player, which, in addition to the limited available set of actions, makes limit-based games significantly smaller than their no-limit counterparts.
On each round, players combine their private cards with the community cards to form the best possible 5-card poker hand, which could include 0, 1, or 2 private cards.

#### Example Texas Hold'em Poker Hand


### Summary of Games


## 2. History of Solving Poker
In 1951, John Nash wrote: "The study of n-person games for which the accepted ethics of fair play imply non-cooperative playing is, of course, an obvious direction in which to apply this theory. And poker is the most obvious target. The analysis of a more realistic poker game than our very simple model should be an interesting affair."

A main interest in games like poker comes from the fact that it is a game of imperfect information. Unlike games of perfect information (e.g. chess) where all information is visible to all players, in poker there is hidden information (opponent player private cards). 

The history of poker solving techniques have gone from rule/formula based to simulation based to, most recently, game theoretical. Game theoretical techniques have evolved by (a) using Monte Carlo techniques so that we don't need to traverse the entire game tree to update the game information and strategies and (b) abstraction


A short timeline of the major milestones in computer poker research is given here:

1998: Opponent Modelling in Poker, Billings et al. (Alberta)
A basic rule-based system was developed in 1999 in the University of Alberta Computer Poker Research Group (CPRG), which took an effective hand-strength as input and outputted a (fold, call, raise) probability triple, using a form of basic opponent modelling. 

2000: Abstraction Methods for Game-Theoretic Poker, Shi and Littman

2003: Approximating Game-Theoretic Optimal Strategies for Full-scale Poker, Billings et al. (Alberta)

2005: Optimal Rhode Island Poker, Gilpin and Sandholm (CMU)

2007: Regret Minimization in Games with Incomplete Information, Zinkevich et al (Alberta)
This hugely important paper introduced the Counterfactual Regret Minimization algorithm, which is the main algorithm used today for finding game-theoretic optimal strategies in poker games. 

MCCFR

2008-9: Man vs. Machine Limit Texas Hold'em Competitions

2015: Heads-up Limit Hold'em Poker is Solved, Bowling et al (Alberta)

2015: Brains vs. AI No-Limit Texas Hold'em Competition

2017: DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker (DeepMind)

2017: Libratus and Man vs. Machine Competition

2019: Superhuman AI for Multiplayer Poker, Brown and Sandholm (CMU)


## 3. Game Theory -- Equilibrium and Regret
Let's look at some important game theory concepts before we get into actually solving for poker strategies. 

What does it mean to "solve" a poker game? In the 2-player setting, this means to find a Nash Equilibrium strategy for the game. If both players are playing this strategy, neither would want to change to a different strategy, since neither could do better with any other strategy. 

Intuition for this in poker can be explained using a simple all-in game where one player must either fold or bet all his chips and the second player must either call or fold if the first player bets all the chips. In this scenario, the second player may begin the game with a strategy of calling a low percentage of hands. After seeing the first player go all-in very frequently, he may increase that percentage. This could lead the first player to reduce his all-in percentage. Once the all-in percentage and the call percentage stabilize such that neither player can unilaterally change his strategy to increase his profit, then the equilibrium strategies have been reached.

But what if the opponent, for example, keeps calling this low percentage of hands and seems to be easy to exploit? The game theoretic solution would not fully take advantage of this opportunity. The best response strategy is the one that maximally exploits the opponent by always performing the highest expected value play against their strategy (and an exploitative strategy is one that exploits an opponent's non-equilibrium play). However, this strategy can itself be exploited. In the above example, this could mean raising all hands after seeing the opponent calling with a low percentage of hands.


### Rock Paper Scissors
We can also think about this concept in Rock-Paper-Scissors. The game matrix for the game is shown below:

| P1/2  | Rock  | Paper  | Scissors  |
|---|---|---|---|
| Rock  | 0, 0  | -1, 1  | 1, -1  |
| Paper  | 1, -1  | 0, 0  | -1, 1  |
| Scissors  | -1, 1  | 1, -1  | 0, 0  |

The equilibrium strategy is to play each action with 1/3 probability. If you deviate from this strategy, you can be exploited by your opponent always playing the action that beats your most favored action. For example, if you play Rock 50%, Paper 25%, and Scissors 25%, your opponent can play Paper 100%. He will win half the time, draw half the time, and lose half the time, resulting in an average profit per game of 1*0.5 + 0*0.25 + (-1)*0.25 = 0.25. 

If your opponent plays the equilibrium strategy of Rock 1/3, Paper 1/3, Scissors 1/3, then he will have the following EV. EV = 1*(1/3) + 0*(1/3) + (-1)*(1/3) = 0. t


| P1/P2  | Rock 50%  | Paper 25% | Scissors 25% |
|---|---|---|---|
| Rock 0%  | 0  | 0  | 0  | 
| Paper 100%  | 0.5*1 = 1  | 0.25*0 = 0  | 0.25*(-1) = -0.25  | 
| Scissors 0%  | 0  | 0  | 0  |



| P1/P2  | Rock 50%  | Paper 25%  | Scissors 25%  |
|---|---|---|---|
| Rock 1/3  | 0  |   |   |
| Paper 1/3  |   |   |   |
| Scissors 1/3  |   |   |   |


test

### Regret
When I think of regret in poker, the first thing that comes to mind is often "Wow you should've played way more hands in 2010 when poker was so easy". Others may regret big folds or bluffs or calls that didn't work out well. 

Here we will look at the mathematical concept of regret. Regret is a measure of how well you could have done compared to some alternative. We can give another example from Rock Paper Scissors: 

If you picked R and your opponent picked 

### Bandits

RPS -- Automation with Winston
Exploitation vs. equilibrium (maybe RPS example)
Bandits

$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$

## 4. Solving Toy Poker Games from Normal Form to Extensive Form

## 5. Trees in Games

## 6. Toy Games and Python Implementation

## 7. Counterfactual Regret Minimization (CFR)

## 8. Game Abstraction

## 9. Agent Evaluation

## 10. CFR Advances

## 11. Deep Learning and Poker

## 12. AI vs. Humans -- What Can Humans Learn?

## 13. Multiplayer Games

## 14. Opponent Exploitation

## 15. Other Games (Chess, Atari, Go, Starcraft, Hanabi)
