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
