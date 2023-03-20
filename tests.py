from aic.parameter import parameter
from aic.poi import tanh_poi, linear_poi
from aic.agent import agent
from random import random

def init_fn(n):
    return n + 2*random()-1, n + 2*random()-1


class test1(parameter):
    param_idx = 0  # Makes it easy to differentiate results by parameter set
    desc="test1"
    
    n_agents = 1
    battery = 20
    time_steps = 50
    speed = 2.0
    map_size = 30

    poi_pos = [[1, 1], [20, 5], [25, 25], [7, 28],[13,28],[16,4],[27,14],[2,13]] 
    n_pois = len(poi_pos)
    poi_class = [tanh_poi] * 4 + [linear_poi] * 4

    agent_class = [agent] * n_agents
    agent_pos = [[15.0,15.0]]

    interact_range = 2.0
    n_sensors = 4

class test2(parameter):
    param_idx = 1  # Makes it easy to differentiate results by parameter set
    desc="test2"
    
    n_agents = 6
    battery = 5
    time_steps = 50
    speed = 2.0
    map_size = 30

    poi_pos = [[1, 1], [20, 5], [25, 25], [7, 28]] 
    n_pois = len(poi_pos)
    poi_class = [tanh_poi] * 2 + [linear_poi] * 2

    agent_class = [agent] * n_agents
    agent_pos = [init_fn(15) for i in range(n_agents)]

    interact_range = 2.0
    n_sensors = 4