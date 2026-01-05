from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from markov import MarkovChain
import sys


class gramPrintListener(gramListener):

    def __init__(self):
        self.trans = {}
        self.trans_noact = {}


    def enterStatesNoRew(self, ctx):
        self.states = [str(x) for x in ctx.ID()]

        self.rewards =None
        print("States: %s" % str(self.states))

    def enterStatesRew(self, ctx):
        self.states = [str(x) for x in ctx.ID()]
        ints=[int(str(x)) for x in ctx.INT()]

        self.rewards=ints
        print("States: %s" % str(self.states))
        print("Rewards: %s" % str(self.rewards))
    

    def enterDefactions(self, ctx):
        self.actions = [str(x) for x in ctx.ID()]
        print("Actions: %s" % str(self.actions))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        if act in self.trans.keys():
            self.trans[act] += [(dep, ids[i], weights[i]) for i in range(len(ids))]
        else:

            self.trans[act] = [(dep, ids[i], weights[i]) for i in range(len(ids))]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        if "" in self.trans.keys():
            self.trans[""] += [(dep, ids[i], weights[i]) for i in range(len(ids))]
        else:
            self.trans[""] = [(dep, ids[i], weights[i]) for i in range(len(ids))]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))



def main():
    input_file=sys.argv[1]
    lexer = gramLexer(FileStream(input_file, encoding='utf-8'))
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    markov=MarkovChain(list_states=printer.states, list_actions= printer.actions, list_rewards= printer.rewards, dict_trans=printer.trans)
    markov.__repr__()
    print(markov.chain)
    # a,b,c=markov.simulation_MC(10)
    # print(a,b,c)
    chosen_end_states=['S3']
    minmax=-1
    
    print(markov.get_initial_states_MC(chosen_end_states))
    print(markov.compute_accessibility_prob_iterative_MDP(2,chosen_end_states))
    print(markov.compute_expected_reward())

    #print(markov.SMC_qualitatif(['S5'], 5, 0.01, 0.01, 0.32, 0.02))

if __name__ == '__main__':
    main()