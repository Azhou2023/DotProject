from operator import truediv
import statistics
from sys import builtin_module_names
from turtle import window_width
import pygame
import neat
import time
import os
import random
pygame.font.init()


white = (255,255,255)
red = (255,0,0)
black = (0,0,0)

font = pygame.font.SysFont("Calibri", 35)
STAT_FONT = pygame.font.SysFont("Calibri", 35)


WIN_WIDTH = 500
WIN_HEIGHT = 700


class Dot:
    def __init__(self):
        self.x = 250
        self.y = 690
        self.height = 10
        self.width = 10
        self.vel = 0
    
    def move(self):
        if self.vel>10:
            self.vel = 10
        if self.vel<-10:
            self.vel = -10
        self.x += self.vel
        
    def moveUp(self):
        self.y-=5
    
    def accLeft(self):
        self.vel -= 1
    
    def decLeft(self):
        if self.vel<0:
            self.vel+=1
        
    def accRight(self):
        self.vel += 1
        
    def decRight(self):
        if self.vel>0:
            self.vel-=1
            
    def draw(self, win):
        pygame.draw.rect(win, black, (self.x, self.y, self.width, self.height))
        
class Obstacle: 
    gap = 50
    passed = False
    def __init__(self, y):
        self.widthOfFirst = random.randrange(75, 375)
        self.height = 30
        self.y = y
    
    def move(self):
        self.y += 5
        
    def draw(self, win):
        pygame.draw.rect(win, red, (0, self.y, self.widthOfFirst, self.height))
        pygame.draw.rect(win, red, (self.widthOfFirst+self.gap, self.y, WIN_WIDTH-(self.widthOfFirst+self.gap), self.height))
    
    
            

def draw_window(win, dots, obstacles, score):
    win.fill(white)
    # e.move(10, 0)
    for d in dots:
        d.draw(win)
    for o in obstacles:
        o.draw(win)
    
    text = STAT_FONT.render("Score: " + str(score), 1, black)
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    pygame.display.update()

def main(genomes, config):
    dots = []
    obstacles = [Obstacle(500),  Obstacle(300),  Obstacle(100),  Obstacle(-100), ]
    nets = []
    ge = []
    clock = pygame.time.Clock()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    score = 0

    obstacle_ind = 0

    
    for _, g in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(g, config))
        dots.append(Dot())
        g.fitness = 0
        ge.append(g)
    
    run = True
    while run:
        score += 1
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
    
    
        obstacle_passed = False
    
        if obstacles[0].y>WIN_HEIGHT:
            obstacles.pop(0)
            obstacles.append(Obstacle(-100))
        
        if dots[0].y>500:
            for dot in dots:
                dot.moveUp()
        else:
            for o in obstacles:
                o.move()
  
                    
        for j, d in enumerate(dots):
            if(d.x < 0 or d.x+d.width>WIN_WIDTH or d.y < 0 or d.y+d.height > WIN_HEIGHT):
                dots.pop(j)
                ge.pop(j)
                nets.pop(j)
            elif ((d.x < obstacles[obstacle_ind].widthOfFirst and  d.y + d.height > obstacles[obstacle_ind].y and d.y < obstacles[obstacle_ind].y + obstacles[obstacle_ind].height) or (d.x + d.width > obstacles[obstacle_ind].widthOfFirst+obstacles[obstacle_ind].gap and d.y + d.height > obstacles[obstacle_ind].y and d.y < obstacles[obstacle_ind].y + obstacles[obstacle_ind].height)):
                dots.pop(j)
                ge.pop(j)
                nets.pop(j)
        
        if len(dots)==0:
            run = False
            break
                
        if dots[0].y < obstacles[obstacle_ind].y:
            obstacle_passed = True
        
        # print("Obstacle y:", obstacles[obstacle_ind].y)
        # print("Dot y:", dots[0].y)
        # print("Boolean:", obstacle_passed)
            
        for x, dot in enumerate(dots): 

            output = nets[x].activate((dot.x, abs(dot.x-obstacles[obstacle_ind].widthOfFirst), abs(dot.x-(obstacles[obstacle_ind].widthOfFirst+obstacles[obstacle_ind].gap))))
            if output[0] > 0:
                dot.accRight()
            elif output[0]<=0:
                dot.decRight()
            if output[1] > 0:
                dot.accLeft()
            elif output[1]<=0:
                dot.decLeft()
            dot.move()
            ge[x].fitness += 0.1
            
            
        if len(dots)!=0 and dots[0].y<obstacles[0].y:
            obstacle_ind = 1

        if obstacle_passed == True:
            for g in ge:
                g.fitness += 20
            obstacle_passed = False
 
    
        draw_window(win, dots, obstacles, score)
    

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main,100)
    print('\nBest genome:\n{!s}'.format(winner))



if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)