# Diplomacy

This is my NLP class project. It is an attempt to take the work of Niculae, Kumar, Boyd-Graber, and Danescu-Niculescu-Mizil in
[Linguistic Harbingers of Betrayal](https://vene.ro/betrayal/#data). This paper discusses the linguistic cues that precede betrayal in a popular
online board game. But despite the authors finding statistically significant differences between message exchanges between dyads destined to betrayal
and dyads who do not end up in betrayal, the classifier that they trained to detect imminent betrayal only managed to attain 57% accuracy. I hope to
take the features that the authors describe as significant and train different classifiers on them to attain better accuracy.

## What's different

### What is different between what I am doing and what the authors in this paper did?

The authors only tried a logistic regression model, and only played around with its hyperparameters a bit. They didn't really make the classifier the centerpiece of their
work (because it wasn't - the features were). This project on the other hand, is all about whether it is indeed feasible to use the features that the authors identified
to detect imminent betrayal. Essentially, <b>the authors of this paper came up with linguistic features that can be used to detect betrayal, but they failed to train
a classifier that convinces me; my project is about taking their work and seeing if I can really train a classifier that works</b>, and along the way, I will explore some
additional features perhaps. I will also collect a new batch of data to attempt validation on a set of data collected in a different context to see if whatever classifier
I make generalizes well.

## Usage

This project aims to predict whether a betrayal is "about" to happen in the game of Diplomacy between two players. By "about" I mean that the program will give a confidence
score to betrayal/not betrayal (and which person in a dyad is going to betray) at different times: i.e., will X betray Y in the next 10 turns? 5 turns? 1 turn?

The ideal usage <b>will be</b> (it isn't implemented yet):

```bash
python3 betrayal.py msg_pairs_one.yml msg_pairs_two.yml msg_pairs_n.yml
```

This will output a matrix like this:

```
###########################################################################################
|  Player  |  Betrays within 10 turns  | Betrays within 5 turns  | Betrays within 1 turn  |
###########################################################################################
|  Bob     |  40%                      |  35%                    |  23%                   |
-------------------------------------------------------------------------------------------
|  Alice   |  13%                      |  4%                     |  2%                    |
-------------------------------------------------------------------------------------------
```

### Message Format

It is important to adhere to the right format for the messages that you feed into the betrayal.py script. There is some metadata that is necessary.
Each file is a series of messages with some metadata from a single season in the game of Diplomacy and for a single pair of people. The series of files that
you feed the program <b>must all be between the same two players</b>. To get best results, you should give the program all of the messages that have so far
occurred between two players.

Here is an example file:

```yaml
year: 1901                                  # <-- This is the year that is ABOUT to be played, not the one that was just played
season: Winter                              # <-- This is the season that is ABOUT to be played, not the one that was just played
message: >
  Hey Bob, this is Alice. Could you destroy everyone else for me, please?
  Thanks
```

