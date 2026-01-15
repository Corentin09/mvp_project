import tkinter as tk
from tkinter import ttk, PhotoImage


class Interface:
    """Create an interface to let the user interact with the simulation of the Markov Chain or the Markov Decision Process
    """

    def __init__(self, markov):
        # ----------- Generic Variable -----------
        self.minmax = 1 #1 if the interface show the max probability for accessibility, else -1 for min 
        self.tot_reward = 0
        self.g, self.u, self.r = None, None, None

        # ----------- Interface Structure -----------
        self.root = tk.Tk()
        self.root.title("MDP")

        self.frame_graph = tk.Frame(self.root)
        self.frame_graph.pack(side="left", padx=10, pady=10)

        self.frame_controls = tk.Frame(self.root)
        self.frame_controls.pack(side="right", padx=10, pady=10)
        self.label_img = tk.Label(self.frame_graph)
        self.label_img.pack()

        self.actions_frame = tk.Frame(self.frame_controls)
        self.actions_frame.pack(pady=10)

        self.graph = markov
        self.history = [self.graph.current_state]

        # ----------- Rewards Labels -----------
        if self.graph.check_rewards():
            self.reward_label = tk.Label(
                self.frame_controls,
                text="",
                justify="left",
                wraplength=300
            )
            self.reward_label.pack(pady=10)

            if self.graph.check_MC(): # show the average reward only for MC
                avg_reward, _ = self.graph.get_average_reward(30, 0.01, 0.01)
                self.avg_reward_label = tk.Label(
                    self.frame_controls,
                    text="",
                    justify="left",
                    wraplength=300
                )
                self.avg_reward_label.pack(pady=10)
                self.avg_reward_label.config(text=f"Récompense Moyenne: {round(avg_reward, 2)}")

        # ----------- Selection of Final and Current State -----------

        self.selected_target = tk.StringVar()

        ttk.Label(self.frame_controls, text="État final :").pack()
        self.target_menu = ttk.Combobox(
            self.frame_controls,
            textvariable=self.selected_target,
            values=[""] + list(self.graph.states),
            state="readonly"
        )
        self.target_menu.pack(pady=5)
        

        self.target_menu.bind("<<ComboboxSelected>>", lambda e: self.analyze_target())
        ttk.Label(self.frame_controls, text="Sélectionner un état :").pack()

        self.selected = tk.StringVar()
        self.current_menu = ttk.Combobox(
            self.frame_controls,
            textvariable=self.selected,
            values=list(self.graph.states),
            state="readonly"
        )
        self.current_menu.pack(pady=5)
        

        self.current_menu.bind("<<ComboboxSelected>>", lambda e: self.change_current())

        # ----------- Minmax button for MDP -----------

        if not self.graph.check_MC():
            ttk.Button(
                    self.frame_controls,
                    text="minmax",
                    command=lambda a=0: self.change_minmax()
                ).pack(pady=3, fill="x")

        self.result_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.result_label.pack(pady=10)

        # ----------- Accessibility for MC and MDP -----------

        self.result_it_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.result_it_label.pack(pady=10)

        self.result_min_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.result_min_label.pack(pady=10)

        self.result_qual_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.result_qual_label.pack(pady=10)

        self.result_quan_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.result_quan_label.pack(pady=10)


    def update_interface(self):
        """Update the interface with the pertinent buttons in function of the current state
        """
        path = self.graph.show_graph(self.g, self.u, self.r)
        img = PhotoImage(file=path)
        self.label_img.img = img
        self.label_img.config(image=img)

        for child in self.actions_frame.winfo_children():
            child.destroy()

        list_action = self.graph.get_possible_actions(self.graph.current_state)
        if "" in list_action:
            list_action[list_action.index("")] = "Next State"
            
        for action in list_action:
            ttk.Button(
                self.actions_frame,
                text=action,
                command=lambda a=action: self.execute_button(a)
            ).pack(pady=3, fill="x")

    def execute_button(self, a):
        """Exectue a step of the simulation, depending of which button is pressed

        Args:
            a (str): The action chosen by the user for the next step
        """
        self.g , self.u, self.r = None, None, None
        self.graph.next_state(a)
        self.update_interface()
        self.history.append(self.graph.current_state)
        self.target_menu.current(0)
        self.result_label.config(text="")
        self.result_min_label.config(text="")
        self.result_quan_label.config(text="")
        self.result_qual_label.config(text="")
        self.tot_reward += self.graph.rewards_dict[self.graph.current_state]
        self.reward_label.config(text=f"Récompense totale: {self.tot_reward}")

    def change_minmax(self):
        """change minmax value to the opposite
        """
        self.minmax = self.minmax*-1
        self.analyze_target()

    def change_current(self):
        """update the current state when a new current state is chosen"""
        self.graph.current_state = list(self.graph.states)[self.current_menu.current()]
        self.analyze_target()

        
    def analyze_target(self):
        """compute accessibility probabilities and initial states, and update the interface with relevant informations
        """
        target = self.selected_target.get()
        if not target or target not in self.graph.states or target=="":
            self.g , self.u, self.r = None, None, None
            self.update_interface()
            self.result_label.config(text="")
            self.result_it_label.config(text="")
            self.result_min_label.config(text="")
            self.result_quan_label.config(text="")
            self.result_qual_label.config(text="")
            self.result_qual_label.config(text="")
            return
        if self.graph.check_MC():
            self.g, self.u, self.r = self.graph.get_initial_states_MC([target])
        else:
            self.g, self.u, self.r = self.graph.get_initial_states_MDP([target], self.minmax)

        self.update_interface()
        if self.graph.check_MC():
            prob = self.graph.compute_accessibility_prob_linear_MC([target])[self.graph.states.index(self.graph.current_state)]
            prob_it = self.graph.compute_accessibility_prob_iterative_MDP(20, [target])[0]

            theta = 0.10
            res_qual = self.graph.SMC_qualitatif([target], 20, 0.01, 0.01, theta, 0.02)
            res_quan = self.graph.SMC_quantitatif([target], 20, 0.01, 0.01)
            

            txt = (
            f"Probabilité d'y arriver depuis "
            f"{self.graph.current_state} : {prob*100}%"
            )
            self.result_label.config(text=txt)

            txt = (
            f"Probabilité d'y arriver depuis "
            f"{self.graph.current_state} : {prob_it*100}%"
            )

            self.result_it_label.config(text=txt)
            if res_qual[0]:
                txt = f"L'état {target} est atteint par analyse qualitative pour theta={100*theta}%"
            else:
                txt = f"L'état {target} n'est pas atteint par analyse qualitative pour theta={100*theta}%"
            self.result_qual_label.config(text=txt)

            print(res_quan)
            if res_quan[0] > 0:
                txt = f"L'état {target} est accessible par analyse quantitative avec une probabilité {100*res_quan[0]}%"
            else:
                txt = f"L'état {target} n'est pas atteint par analyse quantitative"
            self.result_quan_label.config(text=txt)
        else:
            prob_max = self.graph.compute_accessibility_prob_MDP([target], 1)[self.graph.states.index(self.graph.current_state)]
            prob_min = self.graph.compute_accessibility_prob_MDP([target], -1)[self.graph.states.index(self.graph.current_state)]



            txt = (
                f"Probabilité maximale d'y arriver depuis "
                f"{self.graph.current_state} : {prob_max*100}%"
            )

            self.result_label.config(text=txt)

            txt = (
                f"Probabilité minimale d'y arriver depuis "
                f"{self.graph.current_state} : {prob_min*100}%"
            )
            
            self.result_min_label.config(text=txt)



    def execute(self):
        self.update_interface()
        self.root.mainloop()
        return self.history