from __future__ import annotations
import random as rd
import ast
import numpy as np


class MarkovChain():

    def __init__(self, list_states: list[str], list_rewards: list[int]|None = None,  list_actions: list[str] | None = None, dict_trans: dict[str, list[tuple[str,str,int]]] | None = None):
        """ Creates a Markov chain or decision process from a list of states, a list of actions, and a dictionnary of transactions grouped by action"""
        
        self.n = len(list_states)
        self.states = list_states

        # Building the list of actions used
        if list_actions:
            for a in list_actions:
                #removing dummy actions that do not have an associated transition
                if not a in dict_trans.keys():
                    list_actions.remove(a)
            self.actions = list_actions+[""]
        else:
            self.actions=[""]

        #building reward dict
        if list_rewards is not None:
            if len(list_rewards)!=len(list_states):
                raise Exception("Please assign a reward to each and every state")
            self.rewards_dict={self.states[i]: list_rewards[i] for i in range(len(list_rewards))}
        else: 
            self.rewards_dict=None

        self.chain ={a: [[0 for i in range(self.n)] for j in range(self.n)] for a in self.actions}
        self.chain[""]=[[0 for i in range(self.n)] for j in range(self.n)]


        for act in self.chain:

            
            trans = dict_trans[act]
            
            for start, end, w in trans:
                j = self.states.index(start)
                i = self.states.index(end)
                self.chain[act] [j][i]= w 




    def __repr__(self) -> str:
        """prints out the markov chain/process"""
        res=f""
        res+= f"States: {self.states}\n"
        print(f"States: {self.states}")
        res+= f"Actions: {self.actions}\n"
        print(f"Actions: {self.actions}")
        if self.rewards_dict is not None:
            res+= f"Rewards: {self.rewards_dict}"
            print(f"Rewards: {self.rewards_dict}")
        for a in self.actions:
            res+= f"Matrice de l'action {a}: {self.chain[a]}\n"
            print(f"Matrice de l'action {a}: {self.chain[a]}")
        return res



    def get_possible_actions(self, state):
        """Gets all valid actions from a certain state"""
        res=[]
        i = self.states.index(state)
        for a in self.chain:

            for j in range(self.n):
                if self.chain[a][i][j]>0:
                    # there is a valid transaction for action a from i to j
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
        print('last node', last_node)
        for i in range(self.n):
            if self.chain[ast.literal_eval(state_input)][self.states.index(last_node)][i]>0:
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
            case=ast.literal_eval(case)
            state=input("Decision choisie ?")
            while not(self.check_input_is_correct_rule(state, case)):
                case=input("Donner une décision possible")
            controller[case]=state

        return controller
    


    def get_action_from_controller(self, chemin, controller):
        """Looks for the action chosen by the controller by matching the end of the path taken to the sections stored"""
        for i in range(len(chemin)-1, -1, -1):
            if chemin[-i:] in controller:
                return controller[chemin[-i:]]




    def simulation_MC(self, n_transitions: int):
        """goes through the markov chain  for n_transitions, logs then returns its path and its probability"""
        chemin=[self.states[0]]
        tot_prob=1.0
        tot_reward=0
        cur_i=0
        for i in range(n_transitions):
            if self.rewards_dict:
                tot_reward+=self.rewards_dict[chemin[-1]]
            probs=self.chain[""][cur_i]
            chosen_state=rd.choices(self.states, weights=probs, k=1)[0]
            chemin.append(chosen_state)
            cur_i= self.states.index(chosen_state)
            tot_prob*=probs[cur_i]/10
        if self.rewards_dict:
            return chemin, tot_prob, tot_reward
        return chemin, tot_prob, None
    
    def simulation_MDP(self, n_transitions: int, controller: dict[list[str], str]|None = None):
        """ goes through the markov decision process for n_transitions, either making random decisions or asking the user for a controller,
            logs then returns its path and its probability, and the choices made"""
        chosen_method=input(f"Do you want a random choice (answer 1) \n a given controller(answer 2)")
        if int(chosen_method) not in [1,2]:
            chosen_method=input(f"Please choose between 1 and 2")
        match int(chosen_method):
            case 1:
                chemin, choices, proba, tot_reward=self.simulation_MDP_random(n_transitions)
            case 2:
                if controller is None:
                    controller_given=self.ask_for_controller()

                chemin, choices, proba, tot_reward=self.simulation_MDP_controller(n_transitions, controller)
        return chemin, choices, proba, tot_reward


    def simulation_MDP_random(self, n_transitions: int):
        """ goes through the markov decision process for n_transitions making random decisions, 
            logs then returns its path and its probability, and the choices made"""
        chemin=[self.states[0]]
        choices=[]
        tot_prob=1.0
        tot_reward=0
        cur_i=0
        for i in range(n_transitions):
            if self.rewards_dict:
                tot_reward+=self.rewards_dict[chemin[-1]]
            poss_acts=self.get_possible_actions(chemin[-1])
            chosen_act=rd.choice(poss_acts)
            choices.append(chosen_act)
            probs=self.chain[chosen_act][cur_i]
            chosen_state=rd.choices(self.states, weights=probs, k=1)[0]
            chemin.append(chosen_state)
            cur_i= self.states.index(chosen_state)
            tot_prob*=probs[cur_i]/10
        if self.rewards_dict:
            return chemin, choices, tot_prob, tot_reward
        return chemin, choices, tot_prob, None
    

    def simulation_MDP_controller(self, n_transitions:int, controller: dict[list[str], str]|None):
        """ goes through the markov decision process for n_transitions with a controller,
            logs then returns its path and its probability, and the choices made"""
        chemin=[self.states[0]]
        choices=[]
        tot_prob=1.0
        tot_reward=0
        cur_i=0
        for i in range(n_transitions):
            
            if self.rewards_dict:
                tot_reward+=self.rewards_dict[chemin[-1]]
        
            chosen_act=self.get_action_from_controller(chemin, controller)
            
            
            
            choices.append(chosen_act)
            probs=self.chain[chosen_act][cur_i]
            chosen_state=rd.choices(self.states, weights=probs, k=1)[0]
            chemin.append(chosen_state)
            cur_i= self.states.index(chosen_state)
            tot_prob*=probs[cur_i]/10
        if self.rewards_dict:
            return chemin, choices, tot_prob, tot_reward
        return chemin, choices, tot_prob, None


    def check_MC(self):
        if self.actions!=[''] or len(self.chain.keys())>1:
            return False
        return True

    def get_previous_states_MC(self, state):
        i = self.states.index(state)
        res=[self.states[j] for j in range(self.n) if self.chain[''][j][i]>0]
        return res
    

    def get_initial_states_MC(self, end_states):
        guaranteed_states=end_states
        unknown_states=[]

        guaranteed_copy=guaranteed_states.copy()
        unknown_copy=unknown_states.copy()
        is_changed=True
        while is_changed:
            guaranteed_states=guaranteed_copy.copy()
            unknown_states=unknown_copy.copy()
            for s in guaranteed_states+unknown_states:
                for s2 in self.get_previous_states_MC(s):
                    if s2 not in guaranteed_states:
                        i,j=self.states.index(s), self.states.index(s2)
                        if self.chain[""][j][i]==1.0 and s in guaranteed_states:
                            guaranteed_copy.append(s2)
                        else:
                            unknown_copy.append(s2)
            guaranteed_copy=list(set(guaranteed_copy))
            unknown_copy=list(set(unknown_copy))
            is_changed= len(guaranteed_copy)!=len(guaranteed_states) or len(unknown_copy)!=len(unknown_states)
        return guaranteed_states, unknown_states, [s for s in self.states if s not in guaranteed_states and s not in unknown_states]
    

    def get_indices(self, l):
        return [self.states.index(s) for s in l]

    def compute_accessibility_prob_linear_MC(self, end_states):
        guaranteed_states, unknown_states, forbidden_states=self.get_initial_states_MC(end_states)
        guaranteed_indices, unknown_indices, forbidden_indices=sorted(self.get_indices(guaranteed_states)), sorted(self.get_indices(unknown_states)), sorted(self.get_indices(forbidden_states))
        sum_val_l=[sum(self.chain[""][i]) for i in range(self.n)]
        unknown_mat=np.array([[self.chain[""][i][j]/sum_val_l[i] for j in unknown_indices] for i in unknown_indices])
        win_vect=np.zeros(len(unknown_states))
        for i in unknown_indices:
            win_vect[i]=sum([self.chain[""][i][j] for j in guaranteed_indices]) /sum_val_l[i]
        probs_unknown=np.linalg.solve(np.identity(len(unknown_states))-unknown_mat, win_vect)
        res=[0 for i in range(self.n)]
        for i in guaranteed_indices:
            res[i]=1
        for j in unknown_indices:
            res[j]=probs_unknown[unknown_indices.index(j)]
        return res
    
    def compute_accessibility_prob_iterative_MDP(self, n_iter, end_states):
        # TODO complete
        pass
    def compute_accessibility_prob_MDP(self, end_states):
        # TODO complete
        pass

    

    def SMC_quantitatif(self, end_states, n_limit, delta, epsilon):
        if not self.check_MC():
            raise Exception("SMC only with MC")
        n_succ=0
        n_simul=round((np.log(2)-np.log(delta))*(2*epsilon)**(-2))
        for i in range(n_simul):
            chemin, tot_prob, tot_reward=self.simulation_MC(n_limit)
            for valid_state in end_states:
                if valid_state in chemin:
                    n_succ+=1
                    break
        return n_succ/n_simul, n_simul
    
    def SMC_qualitatif(self, end_states, n_limit, alpha, beta, theta, epsilon):
        if not self.check_MC():
            raise Exception("SMC only with MC")
        rm=0
        lima, limb=np.log((1-beta)/alpha), np.log(beta/(1-alpha))
        m=0
        dm=0
        gamma1, gamma0=theta-epsilon, theta+epsilon
        while rm <= lima and rm>=limb:
            chemin, tot_prob, tot_reward=self.simulation_MC(n_limit)
            m+=1
            for valid_state in end_states:
                if valid_state in chemin:
                    dm+=1
                    break
            rm=dm*np.log(gamma1)+(m-dm)*np.log(1-gamma1)-(dm*np.log(gamma0)+(m-dm)*np.log(1-gamma0))
        return rm<=limb, m



        



    

        




    

        


        

        



