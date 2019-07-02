## Intro
This tutorial is made with two target audiences in mind: (1) Those with an interest in poker who want to understand how AI poker agents are made and (2) Those with an interest in coding who want to better understand poker and code poker agents from scratch. 

## Table of Contents
[1. Poker Background](#1-poker-background)<br>
[2. History of Solving Poker](#2-history-of-solving-poker)<br>
[3. Game Theory -- Equilibrium and Regret](#3-game-theory----equilibrium-and-regret)<br>
[4. Solving Toy Poker Games from Normal Form to Extensive Form](#4-solving-toy-poker-games-from-normal-form-to-extensive-form)<br>
[5. Trees in Games](#5-trees-in-games)<br>
[6. Toy Games and Python Implementation](#6-toy-games-and-python-implementation)<br>
[7. Counterfactual Regret Minimization (CFR)](#7-counterfactual-regret-minimization)<br>
[8. Game Abstraction](#8-game-abstraction)<br>
[9. Agent Evaluation](#9-agent-evaluation)<br>
[10. CFR Advances](#10-cfr-advances)<br>
[11. Deep Learning and Poker](#11-deep-learning-and-poker)<br>
[12. AI vs. Humans -- What Can Humans Learn?](#12-ai-vs-humans----what-can-humans-learn)<br>
[13. Multiplayer Games](#13-multiplayer-games)<br>
[14. Opponent Exploitation](#14-opponent-exploitation)<br>
[15. Other Games (Chess, Atari, Go, Starcraft, Dota, Hanabi)](#15-other-games--chess--atari--go--starcraft--hanabi-)<br>

## 1. Poker Background

### Brief History
Poker began at some point in the early 20th century and grew extensively in the early 2000s thanks to the beginnings of online poker, which lead to an accountant and amateur poker player named Chris Moneymaker investing $39 in an online satellite tournament that won him a $10,000 seat at the World Series of Poker Main Event in 2003, alongside 838 other entrants. Moneymaker went on to defeat professional poker player Sam Farha at the final table of the tournament and won $2.5 million. A poker boom was sparked and led to massive player pools on the Internet and in subsequent World Series’ of Poker.

![Chris Moneymaker](https://ucc232ea2ec2f6be5ad4b4d1dbba.dl.dropboxusercontent.com/cd/0/inline/Af-qxPp1gh9fJytr2VJoX0W_fXDx4JQfZAnJ2HqhK8x3F8PBOYcEnSj4-TjDCv6U3U7D6Ihp6juBFWPJcIu-rRWMnu2sT28RyPY4u4EBpnG1RKUtZiWKLz0DVT6TFs7-l9c/file#)
https://www.pokerstarsschool.com/article/Moneymaker-2003-WSOP-ME-Strategy

After the human poker boom, computers also started getting in on the poker action. Researchers began to study solving Texas Hold’em games since 2003, and since 2006, there has been an Annual Computer Poker Competition (ACPC) at the AAAI Conference on Artificial Intelligence in which poker agents compete against each other in a variety of poker formats. In early 2017, for the first time, a NLHE poker agent defeated and is considered superior to top poker players in the world in 1v1 games.

### Basic Rules
Poker is a card game that, in its standard forms, uses a deck of 52 cards composed of four suits (Clubs :clubs:, Diamonds :diamonds:, Hearts :heart:, and Spades :spades:) and 13 ranks (Two through Ten, Jack, Queen, King, and Ace). A dealer button rotates around the table indicating who is the “dealer”. This is decided at random for the first hand and rotates clockwise after that. All actions begin to the left of the hand’s current dealer player.

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
A number of games have been used as artificial intelligence research domains including chess, checkers, Go, and poker, but poker is unique amongst these games because of its key element of imperfect information. In poker, this is the inability to see one’s opponent’s hole cards. Additionally, poker is an excellent research bed for game theoretical research because it has chance events, it is logistically simple yet strategically complex, and it is commonly and competitively played around the world by humans of a variety of skill levels [1]. Finally, it is interesting to see that computers automatically find strategies such as bluffing and slow playing.


## 3. Game Theory -- Equilibrium and Regret
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
