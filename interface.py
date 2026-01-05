import tkinter as tk
from tkinter import ttk, PhotoImage


class Interface:

    def __init__(self, markov):
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

        self.tot_reward = 0
        self.reward_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.reward_label.pack(pady=10)

        self.selected_target = tk.StringVar()

        self.g, self.u, self.r = None, None, None

        ttk.Label(self.frame_controls, text="État final :").pack()
        self.target_menu = ttk.Combobox(
            self.frame_controls,
            textvariable=self.selected_target,
            values=[""] + list(self.graph.states),
            state="readonly"
        )
        self.target_menu.pack(pady=5)
        

        self.target_menu.bind("<<ComboboxSelected>>", lambda e: self.analyze_target())

        self.result_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.result_label.pack(pady=10)

        self.result_min_label = tk.Label(
            self.frame_controls,
            text="",
            justify="left",
            wraplength=300
        )
        self.result_min_label.pack(pady=10)



    def update_interface(self):
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
        self.g , self.u, self.r = None, None, None
        self.graph.next_state(a)
        self.update_interface()
        self.history.append(self.graph.current_state)
        self.target_menu.current(0)
        self.result_label.config(text="")
        self.result_min_label.config(text="")
        self.tot_reward += self.graph.rewards_dict[self.graph.current_state]
        self.reward_label.config(text=f"Récompense totale: {self.tot_reward}")


    def analyze_target(self):
        target = self.selected_target.get()
        if not target or target not in self.graph.states or target=="":
            self.g , self.u, self.r = None, None, None
            self.update_interface()
            self.result_label.config(text="")
            return

        if self.graph.check_MC():
            self.g, self.u, self.r = self.graph.get_initial_states_MC([target])
        else:
            self.g, self.u, self.r = self.graph.get_initial_states_MDP([target])

        self.update_interface()
        if self.graph.check_MC():
            prob = self.graph.compute_accessibility_prob_linear_MC([target])[self.graph.states.index(self.graph.current_state)]

            txt = (
            f"Probabilité d'y arriver depuis "
            f"{self.graph.current_state} : {prob*100}%"
            )

            self.result_label.config(text=txt)
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