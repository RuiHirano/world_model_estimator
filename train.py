from enum import Enum
import enum
import random
from typing import List
from ray.rllib.models.modelv2 import ModelV2
import torch
import torch.nn as nn
import copy
import numpy as np
from torch.optim import SGD, Adam
class Direction(Enum):
    DIRECTION_LEFT = 0
    DIRECTION_RIGHT = 1
    DIRECTION_UP = 2
    DIRECTION_DOWN = 3

class Agent:
    def __init__(self, id) -> None:
        self.id = id
        #self.rate = [[0.8, 0.2], [0.2, 0.8]]
        #self.rate =[[0.9, 0.1], [0.1, 0.9]]
        self.rate =[[1, 0], [0, 1]]
        self.current_rate = self.rate[0]
        self.position = [0, 1, 0, 0, 0]

    def get_action(self, state):
        position_i = state.index(self.id)
        if position_i == 0:
            self.current_rate = self.rate[1]
        elif position_i == len(state) - 1:
            self.current_rate = self.rate[0]

        direction = Direction.DIRECTION_LEFT if random.random() < self.current_rate[0] else Direction.DIRECTION_RIGHT
        return {"id": self.id, "direction": direction}
    
class Agent2:
    def __init__(self, id, actions) -> None:
        self.id = id
        self.actions = actions
        #self.rate = [[0.8, 0.2], [0.2, 0.8]]
        #self.rate =[[0.9, 0.1], [0.1, 0.9]]
        self.rate =[[1, 0], [0, 1]]
        self.current_rate = self.rate[0]

    def get_action(self, state):
        action = random.choice(self.actions)
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

class Env2:
    def __init__(self, agent_map, actions) -> None:
        width = len(agent_map[0])
        height = len(agent_map)
        self.map = [[Cell(str(i)+str(j)) for i in range(width)] for j in range(height)]
        self.actions = actions
        self.step_num = 0
        self.step_data = []
        self.setup(agent_map)

    def setup(self, agent_map):
        for i, line in enumerate(agent_map):
            for k, agents in enumerate(line):
                if agents != None:
                    self.map[i][k].set(agents)


    def step(self):
        new_map = copy.deepcopy(self.map)
        self.step_data = []
        for i, line in enumerate(self.map):
            for k, cell in enumerate(line):
                for agent in cell.agents:
                    action = agent.get_action(self.map)
                    self.step_data.append({"id": agent.id, "action": action, "cell_id": cell.id})
                    print(i, k)
                    if action == Direction.DIRECTION_UP:
                        if i - 1 >= 0:
                            # move left
                            new_map[i-1][k].add(agent)
                            new_map[i][k].delete(agent)
                    elif action == Direction.DIRECTION_DOWN:
                        if i + 1 <= len(self.map) - 1:
                            # move right
                            new_map[i+1][k].add(agent)
                            new_map[i][k].delete(agent)
                    elif action == Direction.DIRECTION_LEFT:
                        if k - 1 >= 0:
                            # move right
                            new_map[i][k-1].add(agent)
                            new_map[i][k].delete(agent)
                    elif action == Direction.DIRECTION_RIGHT:
                        if k + 1 <= len(self.map[0]) - 1:
                            # move right
                            new_map[i][k+1].add(agent)
                            new_map[i][k].delete(agent)
        self.map = new_map
        self.step_num += 1

        return self.map, None, None, None

    def render(self):
        #render_map = [[0] for line in self.map for _ in line]
        #for i, line in enumerate(self.map):
        #    for k, cell in enumerate(line):
        #        print(cell.agents, render_map, i, k, render_map[i])
        #        render_map[i][k] = [agent.id for agent in cell.agents]
        #print(render_map)
        print("Step: {}".format(self.step_num))
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


    def reset(self, agents):
        for agent in agents:
            position_i = random.randint(0, len(self.map)-1)
            self.map[position_i] = agent.id # TODO no duplicate agent at same position
        return self.map

class Env:
    def __init__(self) -> None:
        self.map = [0,0,0,0,0]
        self.actions = [Direction.DIRECTION_LEFT, Direction.DIRECTION_RIGHT]

    def step(self, actions):
        for action in actions:
            index = self.map.index(action["id"])
            if action["direction"] == Direction.DIRECTION_LEFT:
                if index - 1 >= 0 and self.map[index-1] == 0:
                    # move left
                    self.map[index-1] = action["id"]
                    self.map[index] = 0
            elif action["direction"] == Direction.DIRECTION_RIGHT:
                if index + 1 <= len(self.map) - 1 and self.map[index+1] == 0:
                    # move right
                    self.map[index+1] = action["id"]
                    self.map[index] = 0
        return self.map, None, None, None

    def render(self):
        print(self.map)

    def reset(self, agents):
        for agent in agents:
            position_i = random.randint(0, len(self.map)-1)
            self.map[position_i] = agent.id # TODO no duplicate agent at same position
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
    def __init__(self) -> None:
        self.env = Env()
        self.agents = [Agent(id=1)]
        self.data = []
        self.seq_length = 5
        self.batch_size = 16
        self.model = Model(input_dim=self.seq_length, hidden_dim=self.batch_size, output_dim=2)
        self.criterion = nn.MSELoss() #評価関数の宣言
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def correct_data(self, step_num):
        print("correcting data...")
        state = self.env.reset(self.agents)
        for i in range(step_num):
            actions = [agent.get_action(state) for agent in self.agents]
            self.data.append({"state": copy.deepcopy(state), "actions": actions})
            next_state, _, _, _ = self.env.step(actions)
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
                #print(y[-1], y[-1].size(), count)
                #print(x.size(), y.size(), x.dtype) # L, N, H
                output = self.model(x)
                #print(output.size(), y[-1].size())
                label = y[-1]
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                #print(loss.item())
                running_loss += loss.item()
                output_data = torch.tensor(np.eye(2)[torch.argmax(output.data, dim = 1)])
                accuracy += list(np.abs((output_data - label))[:,0]).count(0)
                #print("debug", output_data, label,  accuracy)
            #training_accuracy /= training_size
            accuracy_num = accuracy / len(train_y)
            accuracy_rate = accuracy_num / self.batch_size
            print('epoch: %d loss: %.5f accuracy: %.5f [ %.1f / %d ]' % (e + 1, running_loss, accuracy_rate, accuracy_num, self.batch_size))

            if e % test_epoch_iter == 0:
                self.model.eval()
                accuracy = 0.0
                for x, y in zip(test_x, test_y):
                    output = self.model(x)
                    label = y[-1]
                    output_data = torch.tensor(np.eye(2)[torch.argmax(output.data, dim = 1)])
                    accuracy += list(np.abs((output_data - label))[:,0]).count(0)
                accuracy_num = accuracy / len(test_y)
                accuracy_rate = accuracy_num / self.batch_size
                print('[TEST] epoch: %d accuracy: %.5f [ %.1f / %d ]' % (e + 1, accuracy_rate, accuracy_num, self.batch_size))


        print("finished training")

    def _preprocess(self, data, split=0.8):
        train_data, test_data = data[:int(len(data)*split)], data[int(len(data)*split):]

        def create_data_label(data, seq_length=5, batch_size=16):
            states = [d["state"] for d in data]
            actions = [d["actions"][0]["direction"].value for d in data]
            # N, Seq Length, InputDim
            states = np.array([states[i:i+seq_length] for i in range(0,len(states)-seq_length+1,1)])
            actions = np.array([actions[i:i+seq_length] for i in range(0,len(actions)-seq_length+1,1)])
            actions = np.eye(2)[actions]
            #print("test", np.array(states).shape, np.array(actions).shape)
            # BatchSize, datalen/Batchsize Seq Length, InputDim
            states = np.split(np.array(states[:int(len(states)/batch_size)*batch_size]), batch_size, axis=0)
            actions = np.split(np.array(actions[:int(len(actions)/batch_size)*batch_size]), batch_size, axis=0)
            #print(np.array(states).shape, np.array(actions).shape)

            #  datalen/Batchsize, Seq Length, BatchSize,  InputDim
            states = np.array(states).transpose(1, 2, 0, 3)
            actions = np.array(actions).transpose(1, 2, 0, 3)
            #print(np.array(states).shape, np.array(actions).shape)
            
            return torch.tensor(states, dtype=torch.float), torch.tensor(actions, dtype=torch.float)

        train_x, train_y = create_data_label(train_data, seq_length=self.seq_length, batch_size=self.batch_size)
        test_x, test_y = create_data_label(test_data, seq_length=self.seq_length, batch_size=self.batch_size)
        #print(train_y.size())
        return train_x, train_y, test_x, test_y


def data_test():
    trainer = Trainer()
    trainer.correct_data(30000)
    for d in trainer.data[:100]:
        print("action: {}, state: {}".format(d["actions"][0]["direction"], d["state"]))
    train_x, train_y, test_x, test_y = trainer._preprocess(trainer.data)
    for x, y in zip(train_x[:100], train_y[:100]):
       print("x: {}, y: {}".format(x, y))
    

def env_test():
    env = Env()
    for i in range(10):
        env.step()
        env.render()

def env2_test():

    actions = [Direction.DIRECTION_LEFT, Direction.DIRECTION_RIGHT]
    agent_map = [
        [[Agent2(1, actions)], None, None, None,       None],
    ]

    actions2 = [Direction.DIRECTION_LEFT, Direction.DIRECTION_RIGHT, Direction.DIRECTION_UP, Direction.DIRECTION_DOWN]
    agent_map2 = [
        [[Agent2(1, actions2)], None, None, None,       None],
        [None,       None, None, None,       None],
        [None,       None, None, [Agent2(2, actions2)], None],
        [None,       None, None, None,       None],
        [[Agent2(3, actions2)], None, None, None,       None],
    ]
    env = Env2(agent_map, actions)
    env.render()
    for i in range(10):
        env.step()
        env.render()

def train():
    trainer = Trainer()
    trainer.correct_data(3000)
    trainer.train(epoch=100, test_epoch_iter=10)

if __name__ == "__main__":
    #train()
    env2_test()
