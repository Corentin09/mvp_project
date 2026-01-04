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

    def update_interface(self):
        path = self.graph.show_graph()
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
        self.graph.next_state(a)
        self.update_interface()
        self.history.append(self.graph.current_state)

    def execute(self):
        self.update_interface()
        self.root.mainloop()
        return self.history