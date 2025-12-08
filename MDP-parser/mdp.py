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
        
    def enterDefstates(self, ctx):
        self.states = [str(x) for x in ctx.ID()]
        print("States: %s" % str(self.states))

    def enterDefactions(self, ctx):
        self.actions = [str(x) for x in ctx.ID()]
        print("Actions: %s" % str(self.actions))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.trans[(dep, act)] = [(ids[i], weights[i]) for i in range(len(ids))]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.trans[(dep, "")] = [(ids[i], weights[i]) for i in range(len(ids))]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))



def main():
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    
    markov = MarkovChain(printer.states, printer.actions, printer.trans)
    print(markov)
    markov.print_simulation()

if __name__ == '__main__':
    main()
