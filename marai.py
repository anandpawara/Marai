
import retro
import numpy as np  # For image-matrix/vector operations
import cv2  # For image reduction
import neat
import pickle

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1', record=True)
oned_image = []


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:

       # ob = env.reset  # First image
       # random_action = env.action_space.sample()
       # inx, iny, inc = env.observation_space.shape  # inc = color
        # image reduction for faster processing
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)
        # 20 Networks
        net = neat.nn.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0

        done = False

        
        while not done:

            env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)

            xpos = info['x']

            if xpos >= 65664:
                fitness_current += 10000000
                done = True

            fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward2')
p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
# Save the process after each 10 frames
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
