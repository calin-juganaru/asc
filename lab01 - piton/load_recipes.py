"""
	Bonus task: load all the available coffee recipes from the folder 'recipes/'
	File format:
		first line: coffee name
		next lines: resource=percentage

	info and examples for handling files:
		http://cs.curs.pub.ro/wiki/asc/asc:lab1:index#operatii_cu_fisiere
		https://docs.python.org/3/library/io.html
		https://docs.python.org/3/library/os.path.html
"""

RECIPES_FOLDER = "./recipes/"

def get_recipe(coffee_type):
    recipe_file = RECIPES_FOLDER + coffee_type + ".txt"
    ingredients = {}

    try:
        file = open(recipe_file, "r")
        file.readline()[:-1].lower()

        for line in file:
            aux = line.split('=')
            ingredients[aux[0]] = int(aux[1])

        file.close();

    except FileNotFoundError:
        print("Recipe file %s.txt not found!" % recipe_file);

    return ingredients