""" ASC, ACS, UPB, 20 20 """

import sys
import load_recipes as recipes

# Commands
EXIT = "exit"
LIST_COFFEES = "list"
MAKE_COFFEE = "make"
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
COMMANDS = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]

# Coffee examples
ESPRESSO = "espresso"
AMERICANO = "americano"
CAPPUCCINO = "cappuccino"

# Resources examples
WATER = "water"
COFFEE = "coffee"
MILK = "milk"

# Coffee maker's resources - the values represent the fill percents
RESOURCES = {WATER: 100, COFFEE: 100, MILK: 100}
COFFEES = ["espresso", "americano", "cappuccino"]

# =================================================================================================

def list_coffees():
    """ Print coffee types """
    print(COFFEES, sep=",", end="\n")

def cmd_help():
    """ Print possible commands """
    print(COMMANDS)

def print_status():
    """ Print resources left """
    for resource in RESOURCES:
        print(resource, ': ', RESOURCES[resource])

def make_coffee(coffee):
    """ Makes coffee """
    ingredients = recipes.get_recipe(coffee)

    for resource in RESOURCES:
        if RESOURCES[resource] < ingredients[resource]:
            print("Not enough resources for this coffee ;(")
            return

    for resource in RESOURCES:
        RESOURCES[resource] -= ingredients[resource]

    print("Here's your", coffee)

# =================================================================================================

def main():
    """ The MAIN """

    print("I'm a smart coffee maker")
    print("Enter command:")
    command = sys.stdin.readline().strip("\n").lower()

    while command != EXIT:
        if command == LIST_COFFEES:
            list_coffees()

        if command == HELP:
            cmd_help()

        if command == RESOURCE_STATUS:
            print_status()

        if command == MAKE_COFFEE:
            print("Which coffee?")
            coffee = sys.stdin.readline()[:-1].lower()
            print("How many?")
            cups = int(sys.stdin.readline()[:-1].lower())
            while cups > 0:
                make_coffee(coffee)
                cups = cups - 1

        if command == REFILL:
            print("Which resource? Type 'all' for refilling everything")
            resource = sys.stdin.readline()[:-1].lower()

            if resource == "all":
                for _resource in RESOURCES:
                    RESOURCES[_resource] = 100
            else:
                RESOURCES[resource] = 100

            print_status()

        print("")
        command = sys.stdin.readline()[:-1].lower()

if __name__ == "__main__":
    main()
