from __future__ import annotations
import random as rd
import ast
import graphviz
import numpy as np
from scipy.optimize import linprog


class MarkovChain():

    def __init__(self, list_states: list[str], list_rewards: list[int]|None = None,  list_actions: list[str] | None = None, dict_trans: dict[str, list[tuple[str,str,int]]] | None = None):
        """Initialize a Markov chain or Markov decision process.

        Parameters
        ----------
        list_states : list of str
            Ordered list of state names used by the chain.
        list_rewards : list of int or None, optional
            Rewards associated to each state (same length as list_states). If None,
            no rewards are used. (default: None)
        list_actions : list of str or None, optional
            List of actions available in the decision process. If provided, actions
            not present in dict_trans are removed. The empty string "" is always
            added to represent the default (no-action) transition matrix.
            (default: None)
        dict_trans : dict or None, optional
            Mapping from action name to a list of transitions. Each transition is
            a tuple (start_state, end_state, weight). The function expects an entry
            for each action in self.actions. (default: None)

        Raises
        ------
        Exception
            If list_rewards is provided and its length does not match list_states.

        Notes
        -----
        The transition matrices are stored in self.chain as nested lists where
        self.chain[action][start_index][end_index] == weight.
        """
        self.n = len(list_states)
        self.states = list_states
        self.current_state = self.states[0]

        if len(dict_trans.keys()) > 1 and "" in dict_trans.keys():
            raise Exception("The model has transition with and without actions")

        for a in dict_trans.keys():
            if a not in list_actions and a!="":
                raise Exception(f"The action {a} is not defined")
            
        for trans in dict_trans.values():
            for list_s in trans:
                for s in list_s:
                    if isinstance(s, str) and s not in self.states:
                        print(isinstance(s, str), s not in self.states)
                        raise Exception(f"The state {s} is not defined")
                    
            
            
        

        # Building the list of actions used
        if list_actions:
            for a in list_actions:
                #removing dummy actions that do not have an associated transition
                if a not in dict_trans.keys():
                    list_actions.remove(a)
            self.actions = list_actions
            if "" in dict_trans.keys():
                self.actions += [""]
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
        #self.chain[""]=[[0 for i in range(self.n)] for j in range(self.n)]


        for act in self.chain:

            
            trans = dict_trans[act]
            
            for start, end, w in trans:
                j = self.states.index(start)
                i = self.states.index(end)
                self.chain[act] [j][i]= w 




    def __repr__(self) -> str:
        """Return printable representation of the chain.

        The representation includes states, actions, optional reward mapping and
        the transition matrices for every action. Side-effect: prints the same
        lines to stdout.

        Returns
        -------
        str
            Multi-line string summarizing the chain.
        """
        res=""
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



    def get_possible_actions(self, state: str):
        """Check whether a string represents a non-empty list of valid states.

        The function attempts to parse `input_str` using ast.literal_eval and
        validates that the result is a list of strings, and that each string is
        a known state in this Markov model.

        Parameters
        ----------
        input_str : str
            String to validate (expected to represent a Python list, e.g. "['s1', 's2']").

        Returns
        -------
        bool
            True if `input_str` represents a non-empty list of valid state names,
            False otherwise.
        """
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
        """Check that the controller rule ending can legally perform the action.

        The function checks if the last state in `condition_input` has at least
        one outgoing transition under the action specified by `state_input`.

        Parameters
        ----------
        state_input : str
            Action as a string (literal eval may be used by callers).
        condition_input : list of str
            Sequence of states forming the condition; only the last entry is used.

        Returns
        -------
        bool
            True if the last state in `condition_input` has at least one
            positive-weight outgoing transition under `state_input`; False otherwise.
        """
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
        """Interactively ask the user for a controller mapping.

        The controller is provided through the terminal; each rule is a
        condition (a list of states) mapped to a chosen action. Validation is
        performed to ensure conditions and actions are compatible with the MDP.

        Returns
        -------
        dict
            A mapping from conditions (as lists) to chosen actions (strings).
        """
        n_regles=input("Combien de règles")
        while not n_regles.isnumeric():
            n_regles = input("rendre un nombre entier svp")
        n_regles=int(n_regles)
        controller={}
        for _ in range(n_regles):
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
        """Get action from controller by matching path suffix.

        Searches the controller for the longest rule that matches a suffix of the
        taken path `chemin`. Returns the action associated with the first matching
        suffix starting from the end.

        Parameters
        ----------
        chemin : list of str
            Path (sequence of states) taken so far.
        controller : dict
            Mapping from conditions (represented in the controller) to actions.

        Returns
        -------
        str or None
            Action selected by the controller if a match is found; otherwise None.
        """
        for i in range(len(chemin)-1, -1, -1):
            if chemin[-i:] in controller:
                return controller[chemin[-i:]]




    def simulation_MC(self, n_transitions: int):
        """Simulate the Markov chain for a fixed number of transitions.

        The simulation always starts at the first state in self.states and uses
        the default ('') transition matrix.

        Parameters
        ----------
        n_transitions : int
            Number of transitions to perform.

        Returns
        -------
        tuple
            If rewards are defined: (chemin, tot_prob, tot_reward) where `chemin`
            is the list of visited states (including initial), `tot_prob` is the
            product of the chosen transition probabilities (scaled by /10 in the
            implementation), and `tot_reward` is the sum of rewards collected
            before each transition.
            If no rewards: (chemin, tot_prob, None).
        """
        chemin=[self.states[0]]
        tot_prob=1.0
        tot_reward=0
        cur_i=0
        for _ in range(n_transitions):
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
        """Simulate an MDP with either random or controller-driven choices.

        The function asks interactively whether to perform random decisions or
        use a controller. When controller mode is selected but no controller is
        passed, it will prompt the user to provide one.

        Parameters
        ----------
        n_transitions : int
            Number of transitions to perform.
        controller : dict or None, optional
            Controller mapping used when the controller-driven mode is selected.
            (default: None)

        Returns
        -------
        (chemin, choices, proba, tot_reward)
            chemin : list of str -- visited states (including initial)
            choices : list of str -- actions chosen at each step
            proba : float -- product of probabilities (scaled by /10 in code)
            tot_reward : int or None -- accumulated reward if rewards are defined
        """
        chosen_method=input("Do you want a random choice (answer 1) \n a given controller(answer 2)")
        if int(chosen_method) not in [1,2]:
            chosen_method=input("Please choose between 1 and 2")
        match int(chosen_method):
            case 1:
                chemin, choices, proba, tot_reward=self.simulation_MDP_random(n_transitions)
            case 2:
                if controller is None:
                    controller_given=self.ask_for_controller()

                chemin, choices, proba, tot_reward=self.simulation_MDP_controller(n_transitions, controller)
        return chemin, choices, proba, tot_reward


    def simulation_MDP_random(self, n_transitions: int):
        """Simulate an MDP using random action selection.

        Starting from the first state, at each step a random valid action is
        chosen uniformly from the set of possible actions and then a next state
        is sampled according to that action's transition probabilities.

        Parameters
        ----------
        n_transitions : int
            Number of transitions to perform.

        Returns
        -------
        tuple
            (chemin, choices, tot_prob, tot_reward) where `choices` lists the
            randomly chosen actions and other elements match simulation_MC
            semantics.
        """
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
        """Simulate an MDP using a provided controller.

        At each step the action is selected by querying the controller with the
        current path (using get_action_from_controller), and the next state is
        sampled according to that action's transition probabilities.

        Parameters
        ----------
        n_transitions : int
            Number of transitions to perform.
        controller : dict
            Controller mapping conditions to actions.

        Returns
        -------
        tuple
            (chemin, choices, tot_prob, tot_reward) mirroring other simulation
            methods.
        """
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
    
    def check_rewards(self):
        """Check if the model have rewards

        Returns:
            bool: True if the model has rewards, else False
        """
        return self.rewards_dict is not None


    def check_MC(self):
        """Check whether the model represents a plain Markov chain (no actions).

        Returns
        -------
        bool
            True iff only the default action "" is present.
        """
        if self.actions!=[''] or len(self.chain.keys())>1:
            return False
        return True

    def get_previous_states_MC(self, state):
        """Return states with positive transition probability to `state`.

        Parameters
        ----------
        state : str
            Target state.

        Returns
        -------
        list of str
            States that have a positive-weight transition to `state` under the
            default action.
        """
        i = self.states.index(state)
        res=[self.states[j] for j in range(self.n) if self.chain[''][j][i]>0]
        return res
    

    def get_initial_states_MC(self, end_states):
        """Classify states as guaranteed, unknown or forbidden with respect to reachability.

        This iteratively propagates backward reachability to determine states that
        are guaranteed to reach `end_states`, those that may reach them (unknown),
        and the remainder which cannot reach them.

        Parameters
        ----------
        end_states : list of str
            Target states considered as successful terminal states.

        Returns
        -------
        tuple
            (guaranteed_states, unknown_states, remaining_states)
        """
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
                        if self.chain[""][j][i]==10 and s in guaranteed_states:
                            guaranteed_copy.append(s2)
                        else:
                            unknown_copy.append(s2)
            guaranteed_copy=list(set(guaranteed_copy))
            unknown_copy=list(set(unknown_copy))
            is_changed= len(guaranteed_copy)!=len(guaranteed_states) or len(unknown_copy)!=len(unknown_states)
        return guaranteed_states, unknown_states, [s for s in self.states if s not in guaranteed_states and s not in unknown_states]
    

    def get_indices(self, l):
        """Return indices of a list of states in the internal states list.

        Parameters
        ----------
        l : list of str
            State names.

        Returns
        -------
        list of int
            Indices corresponding to the provided states.
        """
        return [self.states.index(s) for s in l]

    def compute_accessibility_prob_linear_MC(self, end_states):
        """Compute reachability probabilities in an MC by solving linear equations.

        For states that are not trivially guaranteed or forbidden, the method
        sets up linear equations on the unknown states and solves them to get
        the probability of eventually reaching any of `end_states`.

        Parameters
        ----------
        end_states : list of str
            Target states considered successful.

        Returns
        -------
        list of float
            Probability for each state (in order of self.states) to reach any
            of the `end_states`.
        """
        guaranteed_states, unknown_states, forbidden_states=self.get_initial_states_MC(end_states)
        guaranteed_indices, unknown_indices, forbidden_indices=sorted(self.get_indices(guaranteed_states)), sorted(self.get_indices(unknown_states)), sorted(self.get_indices(forbidden_states))
        sum_val_l=[sum(self.chain[""][i]) for i in range(self.n)]
        unknown_mat=np.array([[self.chain[""][i][j]/sum_val_l[i] for j in unknown_indices] for i in unknown_indices])
        win_vect=np.zeros(len(unknown_states))
        for i, idx in enumerate(unknown_indices):
            win_vect[i]=sum([self.chain[""][unknown_indices[i]][j] for j in guaranteed_indices]) /sum_val_l[i]
        probs_unknown=np.linalg.solve(np.identity(len(unknown_states))-unknown_mat, win_vect)
        res=[0 for i in range(self.n)]
        for i in guaranteed_indices:
            res[i]=1
        for j in unknown_indices:
            res[j]=probs_unknown[unknown_indices.index(j)]
        return res

    def get_previous_states_MDP(self, state):
        """Return states with positive transition probability to `state`.

        Parameters
        ----------
        state : str
            Target state.

        Returns
        -------
        list of str
            States that have a positive-weight transition to `state` under the
            default action.
        """
        res = []
        i = self.states.index(state)
        for a in self.chain.keys():
            res=res + [self.states[j] for j in range(self.n) if self.chain[a][j][i]>0]
        return set(res)


    def get_initial_states_MDP(self, end_states: list[str], max: int = 1):
        """
        Classify states as guaranteed, unknown, or forbidden with respect to reachability in an MDP.

        This iteratively propagates backward reachability to determine states that
        are guaranteed to reach `end_states`, those that may reach them (unknown),
        and the remainder which cannot reach them. The `max` parameter determines
        whether to consider the best (max=True) or worst (max=False) choices at each state.

        Parameters
        ----------
        end_states : list of str
            Target states considered as successful terminal states.
        max : int, optional
            If 1, consider the best choices (maximizing reachability).
            If -1, consider the worst choices (minimizing reachability).
            Default is True.

        Returns
        -------
        tuple
            (guaranteed_states, unknown_states, forbidden_states)
        """
        guaranteed_states = set(end_states)
        unknown_states = set()
        forbidden_states = set(self.states) - guaranteed_states

        changed = True
        while changed:
            changed = False
            new_guaranteed = set()
            new_unknown = set()

            for state in self.states:
                if state in guaranteed_states or state in unknown_states:
                    continue

                all_guaranteed = True
                all_possible = True
                min_one_action = False

                for action in self.actions:
                    active_action=False
                    guaranteed_action=True
                    possible_action=False
                    # Check all possible next states for the current action
                    for j in range(self.n):
                        if self.chain[action][self.states.index(state)][j] > 0:
                            active_action=True
                            next_state = self.states[j]
                            
                            if next_state in guaranteed_states:
                                possible_action = True
                            elif next_state in unknown_states:
                                possible_action = True
                                guaranteed_action=False
                            else:
                                guaranteed_action=False
                    if active_action:
                        min_one_action = True
                        if max==1 and guaranteed_action:
                            new_guaranteed.add(state)
                            break
                        elif max==1 and possible_action:
                            new_unknown.add(state)
                            break

                        if max==-1 and not(possible_action):
                            all_possible=False
                        if max==-1 and not(guaranteed_action):
                            all_guaranteed=False
                    
                if max==-1:
                    if all_guaranteed and min_one_action:
                        new_guaranteed.add(state)
                    elif all_possible and min_one_action:
                        new_unknown.add(state)

            if new_guaranteed:
                changed = True
                guaranteed_states.update(new_guaranteed)
                forbidden_states -= new_guaranteed
            if new_unknown:
                changed = True
                unknown_states.update(new_unknown)
                forbidden_states -= new_unknown

        return list(guaranteed_states), list(unknown_states), list(forbidden_states)


    def compute_accessibility_prob_MDP(self, end_states, minmax: int=1):
        """
        Compute reachability probabilities in an MDP by linear programming.

        Parameters
        ----------
        end_states : list of str
            Target states considered successful.
        minmax : int, optional
            +1 (default) to compute the maximum probability of reaching
            `end_states` under an optimal controller; -1 to compute the
            minimum probability (adversarial controller).

        Returns
        -------
        list of float
            A list of length `self.n` giving, for each state in `self.states`,
            the probability (in [0, 1]) of eventually reaching any of `end_states`.
            States classified as guaranteed return 1.0, forbidden return 0.0,
            unknown states are obtained by solving the linear program.

        Raises
        ------
        Exception
            If the linear program fails (scipy.optimize.linprog reports an
            unsuccessful solution) or if internal numerical problems occur.

        Notes
        -----
        The algorithm proceeds in two steps:
        1. Classify states as guaranteed / unknown / forbidden using
           `get_initial_states_MDP`.
        2. For unknown states, build linear inequalities for every state-action
           pair that express reachable mass between unknown states and the
           guaranteed set (normalising per-state action weights). The
           inequalities are arranged and flipped according to `minmax` and
           solved with `scipy.optimize.linprog` to obtain probabilities for
           unknown states.

        Examples
        --------
        >>> mc.compute_accessibility_prob_MDP(['S_goal'], +1)
        [1.0, 0.0, 0.37, 0.89]
        """
        guaranteed_states, unknown_states, forbidden_states=self.get_initial_states_MDP(end_states, minmax)
        guaranteed_indices, unknown_indices, forbidden_indices=sorted(self.get_indices(guaranteed_states)), sorted(self.get_indices(unknown_states)), sorted(self.get_indices(forbidden_states))
        n_unknown=len(unknown_indices)
        if n_unknown==0:
            res=[0 for i in range(self.n)]
            for i in guaranteed_indices:
                res[i]=1
            return res
        ineq_mat=np.zeros((n_unknown*len(self.actions), n_unknown))
        ineq_vect=np.zeros(n_unknown*len(self.actions))


        #computing matrix and vector to solve Ax>=b(or Ax<=b)
        for i in range(n_unknown):# origin state
            for j in range(len(self.actions)):#chosen action

                total=sum(self.chain[self.actions[j]][unknown_indices[i]])

                #checks to avoid division by 0
                if total==0:
                    continue


                line_num=i*len(self.actions)+j
                for k in range(n_unknown):#destination state
                    
                    if k==i:
                        
                        ineq_mat[line_num, k]=-1+self.chain[self.actions[j]][unknown_indices[i]][unknown_indices[k]]/total
                    else:
                        ineq_mat[line_num, k]=self.chain[self.actions[j]][unknown_indices[i]][unknown_indices[k]]/total
                ineq_vect[line_num]=-sum([self.chain[self.actions[j]][unknown_indices[i]][k] for k in guaranteed_indices])/total

        bounds=[[0,1] for i in range(n_unknown)]
        
        #minmax is respectively -1 or +1 to ensure the inequality is the right way
        ineq_mat=minmax*ineq_mat
        ineq_vect=minmax*ineq_vect
        c=[minmax for i in range(n_unknown) ]

        probs_unknown=linprog(c=c, A_ub=ineq_mat, b_ub=ineq_vect, bounds=bounds)

        res=[0 for i in range(self.n)]
        for i in guaranteed_indices:
            res[i]=1
        for j in unknown_indices:
            res[j]=probs_unknown.x[unknown_indices.index(j)]
        return res
    
    def compute_accessibility_prob_iterative_MDP(self, n_iter, end_states):
        guaranteed_states, unknown_states, forbidden_states=self.get_initial_states_MC(end_states)
        guaranteed_indices, unknown_indices, forbidden_indices=sorted(self.get_indices(guaranteed_states)), sorted(self.get_indices(unknown_states)), sorted(self.get_indices(forbidden_states))
        sum_val_l=[sum(self.chain[""][i]) for i in range(self.n)]
        unknown_mat=np.array([[self.chain[""][i][j]/sum_val_l[i] for j in unknown_indices] for i in unknown_indices])
        win_vect=np.zeros(len(unknown_states))
        for i in range(len(unknown_indices)):
            win_vect[i]=sum([self.chain[""][unknown_indices[i]][j] for j in guaranteed_indices]) /sum_val_l[i]
        x=np.zeros(len(unknown_states))
        for i in range(n_iter):
            x=unknown_mat@x+win_vect
        return x
    
    def compute_expected_reward(self):
        assert self.check_MC()
        sum_val_l = [max(sum(self.chain[""][i]), 1e-10) for i in range(self.n)]  # Avoid division by zero
        probability_mat = np.array([[self.chain[""][i][j] / sum_val_l[i] for j in range(self.n)] for i in range(self.n)])
        rew_vect = np.array([self.rewards_dict[self.states[i]] for i in range(self.n)])

        # Add small regularization to diagonal
        epsilon = 1e-10
        expected_rewards = np.linalg.solve(np.identity(self.n) - probability_mat + epsilon * np.identity(self.n), rew_vect)

        return expected_rewards



    def get_average_reward(self, n_limit, delta, epsilon):
        if not self.check_MC():
            raise Exception("SMC only with MC")
        rew = 0
        n_simul=round((np.log(2)-np.log(delta))*(2*epsilon)**(-2))
        for i in range(n_simul):
            chemin, tot_prob, tot_reward=self.simulation_MC(n_limit)
            for s in chemin:
                rew +=self.rewards_dict[s]
                
        return rew/n_simul, n_simul

    def SMC_quantitatif(self, end_states, n_limit, delta, epsilon):
        """Estimate reachability probability via statistical model checking (quantitative).

        Only valid for plain Markov chains (no actions). Uses a fixed number of
        simulations derived from the Chernoff bound.

        Parameters
        ----------
        end_states : list of str
            Target states considered successful.
        n_limit : int
            Number of transitions per simulation.
        delta : float
            Confidence parameter (probability of failure).
        epsilon : float
            Desired additive precision.

        Returns
        -------
        tuple
            (estimated_probability, n_simul) where `estimated_probability` is the
            fraction of runs that hit any of `end_states` within `n_limit`
            transitions, and `n_simul` is the number of simulated trajectories.

        Raises
        ------
        Exception
            If the model is not a Markov chain (i.e. contains actions).
        """
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
        """Perform sequential hypothesis testing for qualitative SMC.

        Uses a sequential probability ratio test (SPRT) to decide whether the
        probability of reaching `end_states` within `n_limit` transitions is
        >= theta+epsilon (accept) or <= theta-epsilon (reject) with prescribed
        error bounds alpha and beta.

        Parameters
        ----------
        end_states : list of str
            Target states considered successful.
        n_limit : int
            Number of transitions per simulation.
        alpha : float
            Type I error rate.
        beta : float
            Type II error rate.
        theta : float
            Threshold probability.
        epsilon : float
            Indifference region width.

        Returns
        -------
        tuple
            (decision, m) where `decision` is True when the test decides in
            favour of probability <= theta-epsilon (accept H0), False otherwise,
            and `m` is the number of samples taken.

        Raises
        ------
        Exception
            If the model is not a Markov chain.
        """
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



    def show_graph(self, garanteed_states: list[str] | None = None, unknown_states: list[str] | None = None, remaining_states: list[str] | None = None):

        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir="LR")

        if garanteed_states is None:
            garanteed_states = []

        if unknown_states is None:
            unknown_states = []

        if remaining_states is None:
            remaining_states = []

        list_node = []
        list_edge = []

        for s in self.states:
            if self.rewards_dict is not None:
                s_node = s + f" ({self.rewards_dict[s]})"
            else:
                s_node = s
            if s in garanteed_states:
                dot.node(s_node, style="filled", fillcolor="green")
            elif s in unknown_states:
                dot.node(s_node, style="filled", fillcolor="orange")
            elif s in remaining_states:
                dot.node(s_node, style="filled", fillcolor="red")
            else:
                if s == self.current_state:
                    dot.node(s_node, style="filled", fillcolor="yellow")
                else:
                    dot.node(s_node, style="filled", fillcolor="lightblue")

            list_node.append(s_node)

        for a in self.actions:
            for i, origin_state in enumerate(list_node):
                for j, destination_state in enumerate(list_node):
                    rate = self.chain[a][i][j]
                    if rate > 0:
                        if a != "":
                            action_node = f"{origin_state}_{a}"
                            edge_a = (origin_state, action_node)
                            edge_dest = (action_node, destination_state)
                            if edge_a not in list_edge:
                                dot.node(action_node, label="", shape="point", width="0.05", height="0.05", fillcolor="black", style="filled")
                                dot.edge(origin_state, action_node, label=f"{a}")
                                list_edge.append(edge_a)

                            dot.edge(action_node, destination_state, label=f"{rate}")
                            list_edge.append(edge_dest)
                        else:
                            dot.edge(origin_state, destination_state, label=f"{rate}")

        output_file = "./mdp"
        dot.render(output_file, cleanup=True)
        return output_file + ".png"
    
    def next_state(self, a):
        if a == "Next State":
            a = ""
        probs = self.chain[a][self.states.index(self.current_state)]
        new_state = rd.choices(self.states, weights=probs)[0]
        self.current_state = new_state


        



    

        




    

        


        

        


