import random

class Truckload:
    def __init__(self):
        self.contents = random.randint(0, 127)

    @classmethod
    def initial_population(cls, qty):
        population = []
        for _ in range(qty):
            population.append(cls())
        return population

class Item(Truckload):
    def __init__(self, name, weight, value):
        super().__init__()
        self.name = name
        self.weight = weight
        self.value = value

def bit_on(n, contents):
    """
    Bitwise shift
    Check if the nth bit in the binary representation of contents is set to 1.
    :return: True if the nth bit is 1, False otherwise.
    """
    return (contents >> n) & 1 == 1

def fitness(contents, item):
    total_weight = 0
    total_value = 0
    
    if bit_on(item, contents):
        total_weight += ITEMS[item].weight
        total_value += ITEMS[item].value

    if total_weight > 4000:
        return 0
    
    return total_value

ITEMS = [
    Item("Cow",       1500,  2000),
    Item("Milk",      1720,   800),
    Item("Cheese",    1000, 12000),
    Item("Butter",    1000,  3000),
    Item("Ice Cream", 1000,  2000),
    Item("Meat",      1200,  8000),
    Item("Leather",   1100,  6000)
]

# Usage
truckload = Truckload()
contents = truckload.contents
for i, item in enumerate(ITEMS):
    item_fitness = fitness(contents, i)
    print(f"{item.name} - Fitness: {item_fitness}")