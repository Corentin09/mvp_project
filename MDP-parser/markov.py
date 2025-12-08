from __future__ import annotations
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np
import random as rd
import ast
import matplotlib.pyplot as plt
import imageio
import typing


class MarkovChain():

    def __init__(self, list_states: list[str], list_actions: list[str] | None = None, dict_trans: dict[tuple[str][str]][list[tuple[str][int]]] | None = None):
        self.n = len(list_states)
        self.states = list_states
        self.actions = [""] + list_actions
        self.chain = {}
        self.list_controller = {}
        for action in self.actions:
            matrix = [[0 for _ in range(self.n)] for _ in range(self.n)]
            for state in self.states:
                i = self.states.index(state)
                if (state, action) in dict_trans.keys():
                    list_res = dict_trans[(state, action)]
                    for state_finish, weight in list_res:
                        j = self.states.index(state_finish)
                        matrix[i][j] = weight
            self.chain[action] = matrix

    def is_mdp(self) -> bool:
        """Determine if the graph is a Markov Chain or a MDP

        Returns:
            bool: Is True if the graph is a MDP, False if it's a Markov Chain
        """
        return len(self.actions) != 1

    def get_possible_actions(self, state):
        """Gets all valid actions from a certain state"""
        res=[]
        i = self.states.index(state)
        for a in self.chain:
            for j in range(self.n):
                if self.chain[a][i][j]>0:
                    # there is a valid transaction for action a
                    res.append(a)
                    break
        return res
    
    
    def check_input_is_state_list(self, input_str: str)->bool:
        """Checks if the input_str is a valid node list coherent with the current MPD"""
        try:
            p=ast.literal_eval(input_str)
            return isinstance(p, list) and all(isinstance(x, str) for x in p) and len(p)>0 and all( x in self.states for x in p)
        except (ValueError, SyntaxError):
            return False

    def check_input_is_correct_rule(self, state_input, condition_input)-> bool:
        """Checks if the last node of the rule can make the state_input decision"""
        last_node=condition_input[-1]
        for i in range(self.n):
            if self.chain[state_input][last_node][i]>0:
                return True
        return False

    def ask_for_controller(self)->dict[list[str], str]:
        """Asks for a controller that is coherent with the current MPD, to be provided through the terminal"""
        n_règles=input("Combien de règles")
        while not n_règles.isnumeric():
            n_règles = input("rendre un nombre entier svp")
        n_règles=int(n_règles)
        controller={}
        for i in range(n_règles):
            case=input("Donner la condition(format ['état1', 'état2'])")
            
            while not(self.check_input_is_state_list(case)):
                case=input("Respecter le format ['état1', 'état2']")
            state=input("Decision choisie ?")
            while not(self.check_input_is_correct_rule(case, state)):
                case=input("Donner une décision possible")
            controller[case]=state
        name_controller = input("Entrez un nom pour le controller")
        if not isinstance(name_controller, str):
            name_controller = f"controller_{len(self.list_controller.keys) + 1}"
        self.list_controller[name_controller] = controller

        return controller
    
    def add_controller(self, controller: dict[list[str], str], name_controller: str | None = None ) -> None:
        if name_controller is None:
            name_controller = f"controller_{len(self.list_controller.keys) + 1}"
        self.list_controller[name_controller] = controller

        return 


    def get_action_from_controller(self, chemin, controller):
        """Looks for the action chosen by the controller by matching the end of the path taken to the sections stored"""
        for i in range(len(chemin)-1, -1, -1):
            if chemin[-i:] in controller:
                return controller[chemin[-i:]]


    # def simulation(self, n_transitions: int, name_controller: int | None = None):
    #     if self.is_mdp():
    #         if 
    #         chemin, choice, proba = self.simulation_MDP(n_transitions=n_transitions)

    def simulation_MC(self, n_transitions: int):
        """goes through the markov chain  for n_transitions, logs then returns its path and its probability"""
        chemin=[self.states[0]]
        tot_prob=1.0
        cur_i=0
        for i in range(n_transitions):
            probs=self.chain[""][cur_i]
            chosen_state=rd.choice(self.states, p=probs)
            chemin.append(chosen_state)
            cur_i= self.states.index(chosen_state)
            tot_prob*=probs[cur_i]/10
        return chemin, tot_prob
    
    def simulation_MDP(self, n_transitions: int, controller: dict[list[str], str]|None = None, is_random: bool = False):
        """ goes through the markov decision process for n_transitions, either making random decisions or asking the user for a controller,
            logs then returns its path and its probability, and the choices made"""
        if is_random:
            chosen_method = 1
        else:
            chosen_method=input(f"Do you want a random choice (answer 1) \n a given controller(answer 2)")
            while int(chosen_method) not in [1,2]:
                chosen_method=input(f"Please choose between 1 and 2")
        match int(chosen_method):
            case 1:
                chemin, choices, proba=self.simulation_MDP_random(n_transitions)
            case 2:
                if controller is None:
                    controller_given=self.ask_for_controller()

                chemin, choices, probs=self.simulation_MDP_controller(n_transitions, controller)
        return chemin, choices, proba


    def simulation_MDP_random(self, n_transitions: int):
        """ goes through the markov decision process for n_transitions making random decisions, 
            logs then returns its path and its probability, and the choices made"""
        chemin=[self.states[0]]
        choices=[]
        tot_prob=1.0
        cur_i=0
        for i in range(n_transitions):
            poss_acts=self.get_possible_actions(chemin[-1])
            print(chemin[-1], poss_acts, self.chain)
            chosen_act=rd.choice(poss_acts)
            choices.append(chosen_act)
            probs=self.chain[chosen_act][cur_i]
            chosen_state=rd.choices(self.states, weights=probs, k=1)[0]
            chemin.append(chosen_state)
            cur_i= self.states.index(chosen_state)
            tot_prob*=probs[cur_i]/10
        return chemin, choices, tot_prob
    

    def simulation_MDP_controller(self, n_transitions:int, controller: dict[list[str], str]|None):
        """ goes through the markov decision process for n_transitions with a controller,
            logs then returns its path and its probability, and the choices made"""
        chemin=[self.states[0]]
        choices=[]
        tot_prob=1.0
        cur_i=0
        for i in range(n_transitions):
            
            chosen_act=self.get_action_from_controller(chemin, controller)
            
            
            
            choices.append(chosen_act)
            probs=self.chain[chosen_act][cur_i]
            chosen_state=rd.choice(self.states, p=probs)
            chemin.append(chosen_state)
            cur_i= self.states.index(chosen_state)
            tot_prob*=probs[cur_i]/10
        return chemin, choices, tot_prob

    def __str__ (self):
        print(type[list(self.states)[0]])
        print(f"States: {self.states}")
        print(f"Actions: {self.actions}")
        for a in self.actions:
            print(f"Matrice de transition pour l'action {a}: {self.chain[a]}")
        return ""
    
    def print_graph(self, name = "Markov_chain.png"):
        "Create a png representing the Markov Chain or the Markov Decision Process"
        G = nx.MultiDiGraph()
        [G.add_node(s, style="filled", fillcolor='white', shape='circle', fixedsize=True, width=0.5) for s in self.states]
        
        labels = {}
        edge_labels = {}
        action_point = " "
        for a in self.actions:
            for i, origin_state in enumerate(self.states):
                show_point = False
                for j, destination_state in enumerate(self.states):
                    rate = self.chain[a][i][j]
                    if rate > 0:
                        if a == "":
                            G.add_edge(origin_state, destination_state, weight=rate, label=rate,len=2)
                        else:
                            if not show_point:
                                action_point += " "
                                G.add_node(action_point, style="filled", fillcolor="black", shape="circle", fixedsize=True, width=0.04, labels=None)
                                G.add_edge(origin_state, action_point, weight=rate, label=f"{a}",len=2)
                                show_point = True
                            
                            G.add_edge(action_point, destination_state, weight=rate, label=rate, len=2)
        #pos = nx.planar_layout(G)
        A = to_agraph(G)
        A.layout()
        A.draw(name)

    def print_simulation(self, name = "Markov_chain.png"):
        "Create a png representing the Markov Chain or the Markov Decision Process"



        G = nx.MultiDiGraph()
        [G.add_node(s) for s in self.states]
        
        labels = {s: s for s in self.states}
        edge_labels = {}
        for a in self.actions:
            for i, origin_state in enumerate(self.states):
                show_point = False
                for j, destination_state in enumerate(self.states):
                    rate = self.chain[a][i][j]
                    if rate > 0:
                        if a == "":
                            G.add_edge(origin_state, destination_state)
                            edge_labels[(origin_state, destination_state)] = rate
                        else:
                            if not show_point:
                                print(type(a))
                                name_node = origin_state + ":" + a
                                G.add_node(origin_state + ":" + a)
                                labels[name_node] = a
                                G.add_edge(origin_state, name_node)
                                show_point = True
                            
                            G.add_edge(name_node, destination_state)
                            edge_labels[(name_node, destination_state)] = rate
        #pos = nx.planar_layout(G)
        pos = nx.spring_layout(G, seed=42)
        color = []
        for node_name, node_label in labels.items():
            if ":" in node_name:
                color.append("white")
            else:
                color.append("blue")
        nx.draw_networkx_nodes(G, pos=pos, node_color=color, label=labels.values())
        nx.draw_networkx_labels(G, pos, labels=labels)
        for node in G:
            out_edges_view = typing.cast(nx.reportviews.OutEdgeView, G.out_edges)
            print(node)
            print(G.edges())
            list_edge = [i for i in G.edges() if i[0] == node]
            print(list_edge)
            if ":" in node:
                nx.draw_networkx_edges(G, pos=pos, edgelist=list_edge, connectionstyle="arc3,rad=0.3")
                nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
            else:
                nx.draw_networkx_edges(G, pos=pos, edgelist=list_edge)
                nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
            #nx.draw_networkx_edge_labels(G, pos)
            

        #nx.draw_networkx(G, pos=pos, arrows=True,node_size=[300, 300, 300, 300, 300] ,node_color= ["blue", "blue", "blue", "white", "white"])
        plt.show()








