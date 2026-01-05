from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from markov import MarkovChain
from interface import Interface
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
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    markov = MarkovChain(list_states=printer.states, list_rewards=printer.rewards, list_actions=printer.actions, dict_trans=printer.trans)
    print(markov.chain)
    interface = Interface(markov)
    history = interface.execute()
    print(history)

if __name__ == '__main__':
    main()