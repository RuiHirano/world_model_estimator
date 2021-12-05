from enum import Enum
import enum
import random
from typing import List, Sequence
from ray.rllib.models.modelv2 import ModelV2
import torch
import torch.nn as nn
import copy
import numpy as np
from torch.optim import SGD, Adam

class Color:
    def __init__(self):
        self.RED = '\033[31m'
        self.GREEN = '\033[32m'
        self.YELLOW = '\033[33m'
        self.END = '\033[0m'
    
    def red(self, text):
        print(self.RED + text + self.END)

    def yellow(self, text):
        print(self.YELLOW + text + self.END)

    def green(self, text):
        print(self.GREEN + text + self.END)

class Action(Enum):
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_UP = 2
    ACTION_DOWN = 3

class Agent:
    def __init__(self, id, actions) -> None:
        self.id = id
        self.actions = actions
        #self.rate = [[0.8, 0.2], [0.2, 0.8]]
        #self.rate =[[0.9, 0.1], [0.1, 0.9]]
        self.rate =[[1, 0], [0, 1]]
        self.current_rate = self.rate[0]

    def get_action(self, state):
        map = state
        width = len(map[0])
        x, y = 0, 0
        for i, line in enumerate(map):
            for k, cell in enumerate(line):
                for agent in cell.agents:
                    if agent.id == self.id:
                        x, y = k, i
                        break

        if x == 0:
            self.current_rate = self.rate[1]
        elif x == width - 1:
            self.current_rate = self.rate[0]
        
        action = Action.ACTION_LEFT if random.random() < self.current_rate[0] else Action.ACTION_RIGHT
        return action

class CollisionAvoidanceAgent:
    def __init__(self, id, actions) -> None:
        self.id = id
        self.actions = actions
        #self.rate = [[0.8, 0.2], [0.2, 0.8]]
        #self.rate =[[0.9, 0.1], [0.1, 0.9]]
        self.rate =[[1, 0], [0, 1]]
        self.current_rate = self.rate[0]

    def get_action(self, state):
        map = state
        height = len(map)
        width = len(map[0])
        x, y = 0, 0
        for i, line in enumerate(map):
            for k, cell in enumerate(line):
                for agent in cell.agents:
                    if agent.id == self.id:
                        x, y = k, i
                        break
        # collision avoidance
        allow_actions = [] 
        if x + 1 <= width -1 and len(map[y][x+1].agents) == 0: # can go right
            allow_actions.append(Action.ACTION_RIGHT)
        if 0 <= x - 1 and len(map[y][x-1].agents) == 0: # can go left
            allow_actions.append(Action.ACTION_LEFT)
        if y + 1 <= height -1 and len(map[y+1][x].agents) == 0: # can go down
            allow_actions.append(Action.ACTION_DOWN)
        if 0 <= y - 1 and len(map[y-1][x].agents) == 0: # can go up
            allow_actions.append(Action.ACTION_UP)
        
        #action = allow_actions[0] if len(allow_actions) > 0 else self.actions[0]
        rate = [0.2, 0.6, 0.1, 0.1]
        sample = random.random()
        action = random.choice(self.actions)
        for i, act in enumerate(self.actions):
            if act in allow_actions:
                r = sum(rate[:i])
                if sample < r:
                    action = act

        #action = random.choice(allow_actions) if len(allow_actions) > 0 else self.actions[0]
        return action

class Cell:
    def __init__(self, id) -> None:
        self.id = id
        self.agents = []
    def set(self, agents):
        self.agents = agents
    def add(self, agent):
        self.agents.append(agent)
    def delete(self, agent):
        new_agents = []
        for ag in self.agents:
            if ag.id != agent.id:
                new_agents.append(ag)
        self.agents = new_agents

class Map:
    def __init__(self, agent_map) -> None:
        self.map, self.width, self.height, self.agent_num = self.setup(agent_map)

    def setup(self, agent_map):
        agent_id_list = []
        width = len(agent_map[0])
        height = len(agent_map)
        map = [[Cell(str(i)+str(j)) for i in range(width)] for j in range(height)]
        for i, line in enumerate(agent_map):
            for k, agents in enumerate(line):
                if agents != None:
                    map[i][k].set(agents)
                    for agent in agents:
                        if agent.id not in agent_id_list:
                            agent_id_list.append(agent.id)
        return map, width, height, len(agent_id_list)

    def get_position(self, id):
        # get agent position
        pass


class Env:
    def __init__(self, agent_map, actions) -> None:
        self.map, self.agent_num = self.setup(agent_map)
        self.old_map = self.map
        self.actions = actions
        self.actions_num = len(actions)
        self.step_num = 0
        self.step_data = []
        self.init_map = self.map

    def setup(self, agent_map):
        agent_id_list = []
        width = len(agent_map[0])
        height = len(agent_map)
        map = [[Cell(str(i)+str(j)) for i in range(width)] for j in range(height)]
        for i, line in enumerate(agent_map):
            for k, agents in enumerate(line):
                if agents != None:
                    map[i][k].set(agents)
                    for agent in agents:
                        if agent.id not in agent_id_list:
                            agent_id_list.append(agent.id)
        return map, len(agent_id_list)


    def step(self):
        new_map = copy.deepcopy(self.map)
        current_map = copy.deepcopy(self.map)
        self.step_data = []
        for i, line in enumerate(self.map):
            for k, cell in enumerate(line):
                for agent in cell.agents:
                    action = agent.get_action(self.map)
                    self.step_data.append({"id": agent.id, "action": action, "cell_id": cell.id})
                    if action == Action.ACTION_UP:
                        if i - 1 >= 0:
                            # move left
                            new_map[i-1][k].add(agent)
                            new_map[i][k].delete(agent)
                    elif action == Action.ACTION_DOWN:
                        if i + 1 <= len(self.map) - 1:
                            # move right
                            new_map[i+1][k].add(agent)
                            new_map[i][k].delete(agent)
                    elif action == Action.ACTION_LEFT:
                        if k - 1 >= 0:
                            # move right
                            new_map[i][k-1].add(agent)
                            new_map[i][k].delete(agent)
                    elif action == Action.ACTION_RIGHT:
                        if k + 1 <= len(self.map[0]) - 1:
                            # move right
                            new_map[i][k+1].add(agent)
                            new_map[i][k].delete(agent)
        self.old_map = current_map
        self.map = new_map
        self.step_num += 1

        return current_map, None, None, self.step_data

    def render(self):
        print("-"*100)
        print("Step: {}".format(self.step_num))
        print("State")
        print(" " + "-"*len(self.old_map[0])*4)
        for i, line in enumerate(self.old_map):
            string = " | "
            for k, cell in enumerate(line):
                if len(cell.agents) > 0:
                    for agent in cell.agents:
                        string += str(agent.id)
                else:
                    string += " "
                string += " | "
            print(string)
            print(" " + "-"*len(self.old_map[0])*4)
        print("Action")
        for data in self.step_data:
            print("id: {}, action: {}, cell: {}".format(data["id"], data["action"], data["cell_id"]))
        print("Next State")
        print(" " + "-"*len(self.map[0])*4)
        for i, line in enumerate(self.map):
            string = " | "
            for k, cell in enumerate(line):
                if len(cell.agents) > 0:
                    for agent in cell.agents:
                        string += str(agent.id)
                else:
                    string += " "
                string += " | "
            print(string)
            print(" " + "-"*len(self.map[0])*4)


    def reset(self):
        self.step_num = 0
        self.step_data = []
        self.map = copy.deepcopy(self.init_map)
        return self.map

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, hidden0=None):
        # input (L, N, H) output: (L, N, D * H)
        output, (hidden, cell) = self.lstm(inputs, hidden0) #LSTM層
        output = self.output_layer(output[-1, :, :]) #全結合層 # (Batch, Output)
        #print(output.size())
        return output

class Trainer:
    def __init__(self, env) -> None:
        self.env = env
        self.data = []
        self.seq_length = 5
        self.input_dim = len(self.env.map) * len(self.env.map[0]) * self.env.agent_num
        self.batch_size = 16
        self.model = Model(input_dim=self.input_dim, hidden_dim=self.batch_size, output_dim=self.env.actions_num)
        self.criterion = nn.MSELoss() #評価関数の宣言
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def correct_data(self, step_num):
        print("correcting data...")
            
        state = self.env.reset()
        for i in range(step_num):
            next_state, _, _, info = self.env.step()
            self.data.append({"state": copy.deepcopy(self._convert_state(state)), "actions": self._convert_actions(info)})
            #self.env.render()
            state = next_state
        print("finished correcting data")

    def train(self, epoch=100, test_epoch_iter=5):
        print("training...")
        train_x, train_y, test_x, test_y = self._preprocess(self.data)

        for e in range(epoch):
            self.model.train()
            running_loss = 0.0
            accuracy = 0.0
            for x, y in zip(train_x, train_y):
                output = self.model(x)
                label = y[-1]
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                output_data = torch.tensor(np.eye(self.env.actions_num)[torch.argmax(output.data, dim = 1)])
                accuracy += list(np.abs((output_data - label))[:,0]).count(0)
            accuracy_num = accuracy / len(train_y)
            accuracy_rate = accuracy_num / self.batch_size
            print('epoch: %d loss: %.5f accuracy: %.5f [ %.1f / %d ]' % (e + 1, running_loss, accuracy_rate, accuracy_num, self.batch_size))

            if e % test_epoch_iter == test_epoch_iter - 1:
                self.model.eval()
                accuracy = 0.0
                for x, y in zip(test_x, test_y):
                    output = self.model(x)
                    label = y[-1]
                    output_data = torch.tensor(np.eye(self.env.actions_num)[torch.argmax(output.data, dim = 1)])
                    accuracy += list(np.abs((output_data - label))[:,0]).count(0)
                accuracy_num = accuracy / len(test_y)
                accuracy_rate = accuracy_num / self.batch_size
                print('[TEST] epoch: %d accuracy: %.5f [ %.1f / %d ]' % (e + 1, accuracy_rate, accuracy_num, self.batch_size))


        print("finished training")

    def _preprocess(self, data, split=0.8):
        train_data, test_data = data[:int(len(data)*split)], data[int(len(data)*split):]
        train_x, train_y = self._create_data_label(train_data, seq_length=self.seq_length, batch_size=self.batch_size)
        test_x, test_y = self._create_data_label(test_data, seq_length=self.seq_length, batch_size=self.batch_size)
        return train_x, train_y, test_x, test_y

    def _create_data_label(self, data, seq_length=5, batch_size=16):
        states = [d["state"] for d in data]
        actions = [d["actions"][0] for d in data] # expect id = 0 agent action
        # N, Seq Length, InputDim
        states = np.array([states[i:i+seq_length] for i in range(0,len(states)-seq_length+1,1)])
        actions = np.array([actions[i:i+seq_length] for i in range(0,len(actions)-seq_length+1,1)])
        actions = np.eye(self.env.actions_num)[actions]
        # BatchSize, datalen/Batchsize Seq Length, InputDim
        states = np.split(np.array(states[:int(len(states)/batch_size)*batch_size]), batch_size, axis=0)
        actions = np.split(np.array(actions[:int(len(actions)/batch_size)*batch_size]), batch_size, axis=0)

        #  datalen/Batchsize, Seq Length, BatchSize,  InputDim
        states = np.array(states).transpose(1, 2, 0, 3)
        actions = np.array(actions).transpose(1, 2, 0, 3)
        
        return torch.tensor(states, dtype=torch.float), torch.tensor(actions, dtype=torch.float)

    def _convert_state(self, state):
        # convert state [Cell, Cell, ...] to [0, 0, 1, 0...]
        onehot_state_list = []
        agent_ids = []
        for i, line in enumerate(state):
            for k, cell in enumerate(line):
                if len(cell.agents) > 0:
                    for ag in cell.agents:
                        if ag.id not in agent_ids:
                            agent_ids.append(ag.id) # add id
                            index = (i + 1) * k
                            onehot_state = [0 for i in range(len(state[0])) for j in range(len(state))]
                            onehot_state[index] = 1
                            onehot_state_list.append({"id": ag.id, "state": onehot_state}) # add state
        # sort onehot state_list by agent ids
        onehot_state_list = sorted(onehot_state_list, key=lambda x: x["id"])
        onehot_state_list = [ data["state"] for data in onehot_state_list] # [[0,1,0,0,0], [0,0,1,0,0]]
        # to 1d [0,1,0,0,0,0,0,1,0,0]
        result = []
        for onehot_state in onehot_state_list:
            result.extend(onehot_state)
        return result

    def _convert_actions(self, info):
        info = sorted(info, key=lambda x: x["id"])
        actions = []
        for data in info:
            action = data["action"].value
            actions.append(action)
        return actions

    def eval(self, step_num=10):
        self.model.eval()
        state = self.env.reset()
        data = []
        accuracy_num = 0
        for i in range(step_num+self.seq_length-1):
            next_state, _, _, info = self.env.step()
            data.append({"state": copy.deepcopy(self._convert_state(state)), "actions": self._convert_actions(info)})
            if len(data) >= self.seq_length:
                eval_x, eval_y = self._create_data_label(data, seq_length=self.seq_length, batch_size=1)
                for x, y in zip(eval_x, eval_y):
                    output = self.model(x)
                    label = y[-1]
                    self.env.render()
                    #print("Predict", output, label, x.size(), x, state)
                    pred_action = self.env.actions[torch.argmax(output[0]).item()]
                    label_action = self.env.actions[torch.argmax(label[0]).item()]
                    print("id: 1, action: label: {}, pred: {}".format(label_action, pred_action))
                    if pred_action == label_action:
                        accuracy_num += 1
                        color.green("Collect Predict")
                    else:
                        color.red("Missed Predict")
                    data.pop(0)
            state = next_state
        print("Accuracy: {:.2f} [ {} / {} ]".format(accuracy_num/step_num, accuracy_num, step_num))

def env_test():
    env = Env(agent_map3, actions3)
    env.render()
    for i in range(10):
        env.step()
        env.render()

def train():
    env = Env(agent_map3, actions3)
    trainer = Trainer(env)
    trainer.correct_data(1000) # correct enough data, if data is less, error may occur.
    trainer.train(epoch=100, test_epoch_iter=5)
    trainer.eval(step_num=100)

if __name__ == "__main__":
    color = Color()
    actions = [Action.ACTION_LEFT, Action.ACTION_RIGHT]
    agent_map = [
        [[Agent(1, actions)], None, None, None,       None],
    ]

    actions2 = [Action.ACTION_LEFT, Action.ACTION_RIGHT, Action.ACTION_UP, Action.ACTION_DOWN]
    agent_map2 = [
        [[Agent(1, actions2)], None, None, None,       None],
        [None,       None, None, None,       None],
        [None,       None, None, [Agent(2, actions2)], None],
        [None,       None, None, None,       None],
        [[Agent(3, actions2)], None, None, None,       None],
    ]

    actions3 = [Action.ACTION_LEFT, Action.ACTION_RIGHT, Action.ACTION_UP, Action.ACTION_DOWN]
    agent_map3 = [
        [[CollisionAvoidanceAgent(1, actions3)], None, None, [CollisionAvoidanceAgent(7, actions3)],       None],
        [None,       None, [CollisionAvoidanceAgent(2, actions3)], None,       None],
        [None,       [CollisionAvoidanceAgent(3, actions3)], None, [CollisionAvoidanceAgent(4, actions3)], None],
        [None,       None, None, None,       None],
        [[CollisionAvoidanceAgent(5, actions3)], None, None, [CollisionAvoidanceAgent(6, actions3)],       None],
    ]
    train()
    #env_test()
