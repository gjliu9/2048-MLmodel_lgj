from game2048.game import Game
from game2048.displays import Display
from exp_imitation_learning.model import RNN

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 20

    N64=0
    N128=0
    N256=0
    N512=0
    N1024=0
    N2048=0

    '''====================
    Use your own agent here.'''
    from game2048.agents import MyRnnAgent as TestAgent
    '''===================='''

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        if score==64:
            N64=N64+1
        elif score==128:
            N128=N128+1
        elif score==256:
            N256=N256+1
        elif score==512:
            N512=N512+1
        elif score==1024:
            N1024=N1024+1
        elif score==2048:
            N2048=N2048+1

        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
    #print(N64,N128,N256,N512,N1024,N2048)
