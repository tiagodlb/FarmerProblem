### import random
#class Truckload:
#    population = []
#    def __init__(self):
#        self.contents = random.randint(0, 127)
#    @classmethod
#    def initial_population(cls, qty):
#        population = []
#        for i in range(qty):
#            population.append(cls())
#        return population
#
#print(Truckload.initial_population(10))

class Truckload:
    def __init__(self,name,weight,value):
        self.name = name
        self.weight = weight
        self.value = value


# Step 1: Initialize
items = [
    #         Name       Weight  Value
    Truckload("Cow",       1500,  2000),
    Truckload("Milk",      1720,   800),
    Truckload("Cheese",    1000, 12000),
    Truckload("Butter",    1000,  3000),
    Truckload("Ice Cream", 1000,  2000),
    Truckload("Meat",      1200,  8000),
    Truckload("Leather",   1100,  6000)
]