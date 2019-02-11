# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents
# Statement
It is a course project and TA provides us with an expectimax agent which can get scores above 2048. Therefore, we can collect board and the agent’s corresponding solution (direction: up, down, left,  right). It’s just like an image classification task where boards are parallel to images, solutions are parallel to labels. Our aim is to make our own agent's score reach 900+. 
Deep learning part: my net: game2048/Model.py, training: game2048/train_CNN.py.
# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances. Added `LXYAgent`
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
    * [`mergedata.py`](game2048/mergedata.py): preprocess the data obtained from ExpectiMax
    * [`Model.py`](game2048/Model.py): My Net
    * [`train_CNN.py`](game2048/train_CNN.py): train my Net
    * [`para.pkl`](game2048/para.pkl): the best parameters for my net
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`get_layer_boards/`](get_layer_boards/): 
    * [`agents.py`](get_layer_boards/agents.py): get data from Expectimax Agent
    * [`getboards.py`](get_layer_boards/getboards.py): get data from Expectimax Agent
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate `LXYAgent`.
* [`generate_fingerprint.py`](generate_fingerprint.py): get `LXYAgent`'s fingerprint.
* [`past_train_log.out`](past_train_log.out): part of my past train log.

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask
* Pytorch

# To train the net
```bash
Before train the net, you should create the dir ./game2048/saved
then:
python3 train_CNN.py
```
# To evaluate LXYAgent
```bash
python3 evaluate.py >>evaluation.log
```
# To generate fingerprint
```bash
python3 generate_fingerprint.py
```
# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

