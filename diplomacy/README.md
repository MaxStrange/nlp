# Diplomacy

This is my NLP class project. It is an attempt to take the work of Niculae, Kumar, Boyd-Graber, and Danescu-Niculescu-Mizil in
[Linguistic Harbingers of Betrayal](https://vene.ro/betrayal/) and build on it. This paper discusses the linguistic cues that precede betrayal in a popular
online board game. But despite the authors finding statistically significant differences between message exchanges between dyads destined to betrayal
and dyads who do not end up in betrayal, the classifier that they trained to detect imminent betrayal only managed to attain 57% accuracy. I trained some classifiers on
the same features and managed to get quite a bit better accuracy out of the models.

## What's different

### What is different between what I am doing and what the authors in this paper did?

The authors only tried a logistic regression model, and only played around with its hyperparameters a bit. They didn't really make the classifier the centerpiece of their
work (because it wasn't - the features were). This project on the other hand, is all about whether it is indeed feasible to use the features that the authors identified
to detect imminent betrayal. Essentially, <b>the authors of this paper came up with linguistic features that can be used to detect betrayal, but they failed to train
a classifier that convinces me; my project is about taking their work and seeing if I can really train a classifier that works</b>. This project is also about building
an NLP system end-to-end - the finished project is able to take messages in YAML format and determine if a betrayal is imminent.

## Usage

This project aims to predict whether a betrayal is "about" to happen in the game of Diplomacy between two players. By this, I mean that the program will output the results of
a bunch of binary classifiers based on the data. These results predict whether betrayal is imminent in the upcoming order resolution phase.

In one terminal:
```bash
cd path/to/this/repo
cd src
./run_server.sh
```
This will run the Stanford CoreNLP server so that the Python code can communicate with it.
Then in another terminal:
```bash
cd path/to/this/repo
cd src
```
Now make sure you have all the python dependencies.
- First, this will only work with Python3 (I haven't tried it with Python2, but I can't imagine that it will work).
- Second, take a look at the requirements.txt file, you CAN `pip3 install -r requirements.txt`, but I don't recommend it - it will install specifically these versions.
There's a good reason for that (these are the versions that work - I can't say that future versions of these libraries will no break backwards compatibility), so I would
recommend setting up a virtual environment for this. Or else just go through the file and make sure you have all those requirements already (with versions greater than or equal to what
is listed) - then try the following. If it doesn't work, it is likely a version issue and you will need to set up a virtual environment. Ah the joy of deploying python programs.

```bash
python3 betrayal.py example_game/1901FallAR.yml example_game/1902SpringAR.yml example_game/1902FallAR.yml
```

This will output what each model believes is likely to happen. (For this example, they all believe there will be no betrayal).

### Message Format

It is important to adhere to the right format for the messages that you feed into the betrayal.py script. There is some metadata that is necessary.
Each file is a series of messages with some metadata from a single season in the game of Diplomacy and for a single pair of people. The series of files that
you feed the program <b>must all be between the same two players</b>. To get best results, you should give the program all of the messages that have so far
occurred between two players.

Here is an example file:

```yaml
year: 1901            # <-- This is the year whose orders are going to be resolved after these messages
season: Spring        # <-- This is the season whose orders are going to be resolved after these messages
game: 19              # <-- Not used, just for your records

a_to_b:
  from_player: Alice  # <-- Not used, just for your records
  from_country: England
  to_player: Bob      # <-- Not used, just for your records
  to_country: Turkey
  messages:
    - >
      Hey Bob, this is Alice. Could you destroy everyone else for me, please?
      Thanks
    - >
      In response to your previous message, no I don't think I can do that - I think France might be upset if I were to take Brest from him.
      We are currently planning on bouncing in the Channel. I know, it is a waste of a move, but otherwise he is just going to go for it.

b_to_a:
  from_player: Bob   # <-- Not used, just for your records
  from_country: Turkey
  to_player: Alice   # <-- Not used, just for your records
  to_country: England
  messages:
    - >
      Hey Alice! Good of you to drop me a line; even though we are far away and unlikely to interact directly until towards the end (assuming we
      both make it that far), I think we could at least offer each other advice. Whadya say? So to start off, I'll give you some advice: go get France!
      He's sneaky. You can't trust him for a moment. I would seriously consider setting up a convoy to his heartland or at least a stab towards Brest.
```

You can look in the example_game directory to see what an entire game looks like coded into this format.


