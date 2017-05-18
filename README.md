Trumpet
===

This is a deep learning character modelling implementation developed for demonstration
purposes for my talk at PyGrunn 2017. It is heavily inspired by Andrej Karpathy's
article "The Unreasonable Effectiveness of Recurrent Neural Networks", but models
Donald Trump tweets rather than Shakespeare.

TODO:

- Define dependencies in a requirements.txt (you basically need Tensorflow + Numpy)
- Add script for generating new tweets based on trained models (sampling functionality is
there, but it only samples during training)


Load and prepare the training data by running `python3 prepare_data.py`.

Train a new modell with `python3 train.py`.

#### Note

Although performance is reasonable, it's far from state of the art. I deliberately
didn't touch things like dropout, batch normalization or other fancy techniques to
keep it as simple as possible for a deep learning how-to presentation. Might improve
things at a later time.
