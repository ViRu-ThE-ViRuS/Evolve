# Evolve

[![CircleCI](https://circleci.com/gh/ViRu-ThE-ViRuS/Evolve.svg?style=svg)](https://circleci.com/gh/ViRu-ThE-ViRuS/Evolve)

### Evolution Strategy for Tensorflow Keras-Models

Evolution Strategy **(ES)** is an optimisation technique based on ideas of adaptation, evolution, mutation, and breeding. It tries to simulate the natural process of selection, and encompasses the survival of the fittest ideaolgy to train an artificial neural network *(a fundamentally different appraoch than Gradient Descent)* using reinforcement learning.

You can read more about it in a paper by [openai](https://blog.openai.com/evolution-strategies/).

This implementation can be used to train any model built using [Keras](https://www.tensorflow.org/guide/keras) api, and [openai/gym](https://github.com/openai/gym) like environments.

#### Installation

From the Github Repository:

```
$ pip install git+https://github.com/ViRu-ThE-ViRuS/Evolve.git
```

##### Dependencies

- use **python3**

- install dependencies from **requirements.txt**:

  ```
  $ pip install -r requirements.txt
  ```

#### Sample Usage

- can be found in **example.py**:

  ```
  $ python example.py
  ```

#### How To Use

- documentation is **in progress**

##### Installation
- run setup:
    ```
    $ ./scripts/setup.sh
    $ source venv/bin/activate
    ```

##### Run Scripts
- run checks:
    ```
    $ ./scripts/check.sh
    ```
- run automated fixes:
    ```
    $ ./scripts/fix.sh
    ```