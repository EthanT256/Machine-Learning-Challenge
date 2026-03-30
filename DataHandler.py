import numpy as np
import pandas as pd
import re
from collections import Counter

def dropColumn(data: pd.DataFrame, columnName: str) -> None:
    data.drop(columns=[columnName], inplace=True)

def rename_columns(data: pd.DataFrame) -> None:
    data = data.rename(columns={
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "Q1",
    "Describe how this painting makes you feel.": "Q2",
    "This art piece makes me feel sombre.": "Q3",
    "This art piece makes me feel content.": "Q4",
    "This art piece makes me feel calm.": "Q5",
    "This art piece makes me feel uneasy.": "Q6",
    "How many prominent colours do you notice in this painting?": "Q7",
    "How many objects caught your eye in the painting?": "Q8",
    "How much (in Canadian dollars) would you be willing to pay for this painting?": "Q9",
    "If you could purchase this painting, which room would you put that painting in?": "Q10",
    "If you could view this art in person, who would you want to view it with?": "Q11", 
    "What season does this art piece remind you of?": "Q12",
    "If this painting was a food, what would be?": "Q13",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "Q14",
    "Painting": "Label"
})

def question_conversions(data: pd.DataFrame) -> None:
    # Q3-Q6:
    data["Q3"] = (data["Q3"].str.split(" - ").str[0].astype(float))
    data["Q4"] = (data["Q4"].str.split(" - ").str[0].astype(float))
    data["Q5"] = (data["Q5"].str.split(" - ").str[0].astype(float))
    data["Q6"] = (data["Q6"].str.split(" - ").str[0].astype(float))
    # Q9:
    data["Q9"] = data["Q9"].str.extract(r'(\d+\.?\d*)').astype(float)
    # Q10-Q12:
    data["10a"] = data["Q10"].apply(lambda x: 1 if "Bedroom" in str(x) else 0)
    data["10b"] = data["Q10"].apply(lambda x: 1 if "Bathroom" in str(x) else 0)
    data["10c"] = data["Q10"].apply(lambda x: 1 if "Office" in str(x) else 0)
    data["10d"] = data["Q10"].apply(lambda x: 1 if "Living room" in str(x) else 0)
    data["10e"] = data["Q10"].apply(lambda x: 1 if "Dining room" in str(x) else 0)

    # Creating indicator features for Q: If you could view this art in person, who would you want to view it with?
    data["11a"] = data["Q11"].apply(lambda x: 1 if "Friends" in str(x) else 0)
    data["11b"] = data["Q11"].apply(lambda x: 1 if "Family members" in str(x) else 0)
    data["11c"] = data["Q11"].apply(lambda x: 1 if "Coworkers/Classmates" in str(x) else 0)
    data["11d"] = data["Q11"].apply(lambda x: 1 if "Strangers" in str(x) else 0)
    data["11e"] = data["Q11"].apply(lambda x: 1 if "By yourself" in str(x) else 0)

    # Creating indicator features for Q: What season does this art piece remind you of?
    data["12a"] = data["Q12"].apply(lambda x: 1 if "Spring" in str(x) else 0)
    data["12b"] = data["Q12"].apply(lambda x: 1 if "Summer" in str(x) else 0)
    data["12c"] = data["Q12"].apply(lambda x: 1 if "Fall" in str(x) else 0)
    data["12d"] = data["Q12"].apply(lambda x: 1 if "Winter" in str(x) else 0)

    # We can drop the original columns:
    data.drop("Q10", axis=1, inplace=True)
    data.drop("Q11", axis=1, inplace=True)
    data.drop("Q12", axis=1, inplace=True)

    # We dont want any entries where the respondent did not answer the question:
    cols_10 = ["10a", "10b", "10c", "10d", "10e"]
    cols_11 = ["11a", "11b", "11c", "11d", "11e"]
    cols_12 = ["12a", "12b", "12c", "12d"]

    data = data[data[cols_10].sum(axis=1) > 0]
    data = data[data[cols_11].sum(axis=1) > 0]
    data = data[data[cols_12].sum(axis=1) > 0]

    data = data.dropna()

# For Q13:
def normalize(text):
    text = str(text).lower().strip()

    # spelling mistakes
    text = text.replace("popscicle", "popsicle")
    text = text.replace("spagetti", "spaghetti")
    text = text.replace("sandwhich", "sandwich")
    text = text.replace("doughnut", "donut")
    text = text.replace("bluberries", "blueberry")
    text = text.replace("macha", "matcha")
    text = text.replace("yoghurt", "yogurt")
    text = text.replace("cantelope", "cantaloupe") 
    text = text.replace("pomegrante", "pomegranate") 
    text = text.replace("fettucine", "fettuccine")

    # common variants
    text = text.replace("blue berry", "blueberry")
    text = text.replace("blueberries", "blueberry")
    text = text.replace("blackberries", "blackberry")
    text = text.replace("raspberries", "raspberry")
    text = text.replace("strawberries", "strawberry")
    text = text.replace("hamburger", "burger")
    text = text.replace("cheeseburger", "burger")
    text = text.replace("cheesecake", "cake")
    text = text.replace("cupcake", "cake")
    text = text.replace("macaron", "cake")
    text = text.replace("cream puff", "cake")
    text = text.replace("brownie", "cake")
    text = text.replace("flan", "cake")
    text = text.replace("panna cotta", "cake")
    text = text.replace("creme brulee", "cake")
    text = text.replace("swiss roll", "cake")
    text = text.replace("croquembouche", "cake") 
    text = text.replace("tiramisu", "cake") 
    text = text.replace("parfait", "cake")  
    text = text.replace("meringue", "cake") 
    text = text.replace("muffin", "cake") 
    text = text.replace("creme brule", "cake") 
    text = text.replace("oreo", "cookie")
    text = text.replace("omelette", "egg")
    text = text.replace("jolly rancher", "candy")
    text = text.replace("lollipop", "candy")
    text = text.replace("carrot", "vegetable")
    text = text.replace("asparagus", "vegetable")
    text = text.replace("cabbage", "vegetable")
    text = text.replace("green peas", "vegetable")
    text = text.replace("boiled leafy greens", "vegetable")
    text = text.replace("sauteed greens", "vegetable")
    text = text.replace("onion", "vegetable")
    text = text.replace("pickle", "vegetable")
    text = text.replace("waffle", "pancake")
    text = text.replace("bagel", "bread")
    text = text.replace("bruschetta", "bread")
    text = text.replace("alcohol", "wine") 
    text = text.replace("whisky", "wine")
    text = text.replace("kool aid", "juice")
    text = text.replace("lemonade", "juice")
    text = text.replace("olive", "fruit")
    text = text.replace("starfruit", "fruit")
    text = text.replace("grapefruit", "fruit")
    text = text.replace("dragonfruit", "fruit")
    text = text.replace("durian", "fruit")
    text = text.replace("coconut", "fruit")
    text = text.replace("olive", "fruit")
    text = text.replace("cantaloupe", "fruit")
    text = text.replace("pear", "fruit")
    text = text.replace("pomegranate", "fruit")
    text = text.replace("pineapple", "fruit")
    text = text.replace("cherry", "fruit")
    text = text.replace("cherries", "fruit")
    text = text.replace("wintermelon", "melon")
    text = text.replace("sardines", "fish")
    text = text.replace("salmon", "fish")
    text = text.replace("tuna", "fish")
    text = text.replace("steak", "meat")
    text = text.replace("lamb", "meat")
    text = text.replace("mutton", "meat")
    text = text.replace("sausage", "meat")
    text = text.replace("prime rib", "meat")
    text = text.replace("filet mignon", "meat")
    text = text.replace("meatloaf", "meat")
    text = text.replace("cold", "iced")
    text = text.replace("cool", "iced")
    text = text.replace("snow", "iced")
    text = text.replace("freezie", "iced")
    text = text.replace("raisin", "grape")
    text = text.replace("french fries", "fry")
    text = text.replace("fries", "fry")
    text = text.replace("icecream", "ice cream")
    text = text.replace("gelato", "ice cream")
    text = text.replace("sorbet", "ice cream")
    text = text.replace("popsicle", "ice cream")
    text = text.replace("sundae", "ice cream")
    text = text.replace("latte", "coffee")
    text = text.replace("hot cocoa", "coffee")
    text = text.replace("teacoffee", "coffee")
    text = text.replace("chowder", "soup") 
    text = text.replace("fettuccine", "pasta")

    # General fixes
    text = text.replace("apples", "apple")
    text = text.replace("asparaguses", "asparagus")
    text = text.replace("avocados", "avocado")
    text = text.replace("bananas", "banana")
    text = text.replace("beans", "bean")
    text = text.replace("beefs", "beef")
    text = text.replace("breads", "bread")
    text = text.replace("broccolis", "broccoli")
    text = text.replace("burgers", "burger")
    text = text.replace("butters", "butter")
    text = text.replace("blueberries", "blueberry")
    text = text.replace("blackberries", "blackberry")

    text = text.replace("candys", "candy")
    text = text.replace("candies", "candy")
    text = text.replace("cakes", "cake")
    text = text.replace("caramels", "caramel")
    text = text.replace("carrots", "carrot")
    text = text.replace("caviars", "caviar")
    text = text.replace("cereals", "cereal")
    text = text.replace("cheeses", "cheese")
    text = text.replace("chickens", "chicken")
    text = text.replace("chocolates", "chocolate")
    text = text.replace("coffees", "coffee")
    text = text.replace("cookies", "cookie")
    text = text.replace("corns", "corn")
    text = text.replace("crabs", "crab")
    text = text.replace("croissants", "croissant")
    text = text.replace("cucumbers", "cucumber")
    text = text.replace("curries", "curry")

    text = text.replace("donuts", "donut")
    text = text.replace("ducks", "duck")
    text = text.replace("durians", "durian")

    text = text.replace("eggs", "egg")

    text = text.replace("fishs", "fish")
    text = text.replace("fishes", "fish")
    text = text.replace("flans", "flan")
    text = text.replace("fries", "fry")
    text = text.replace("fruits", "fruit")

    text = text.replace("garlics", "garlic")
    text = text.replace("gingers", "ginger")
    text = text.replace("geese", "goose")
    text = text.replace("grapes", "grape")
    text = text.replace("grapefruits", "grapefruit")
    text = text.replace("guacamoles", "guacamole")

    text = text.replace("hams", "ham")
    text = text.replace("honeys", "honey")
    text = text.replace("hummuses", "hummus")

    text = text.replace("ice creams", "ice cream")
    text = text.replace("iceds", "iced")

    text = text.replace("jellies", "jelly")
    text = text.replace("juices", "juice")

    text = text.replace("kimchis", "kimchi")
    text = text.replace("kiwis", "kiwi")

    text = text.replace("lambs", "lamb")
    text = text.replace("lasagnas", "lasagna")
    text = text.replace("lemons", "lemon")
    text = text.replace("lentils", "lentil")
    text = text.replace("lettuces", "lettuce")
    text = text.replace("lobsters", "lobster")

    text = text.replace("mangos", "mango")
    text = text.replace("meats", "meat")
    text = text.replace("matchas", "matcha")
    text = text.replace("milks", "milk")
    text = text.replace("mints", "mint")
    text = text.replace("mochis", "mochi")
    text = text.replace("muffins", "muffin")
    text = text.replace("mushrooms", "mushroom")

    text = text.replace("naans", "naan")
    text = text.replace("noodles", "noodle")

    text = text.replace("oatmeals", "oatmeal")
    text = text.replace("octopuses", "octopus")
    text = text.replace("olives", "olive")
    text = text.replace("omelettes", "omelette")
    text = text.replace("onions", "onion")
    text = text.replace("oranges", "orange")

    text = text.replace("pancakes", "pancake")
    text = text.replace("pastas", "pasta")
    text = text.replace("pastries", "pastry")
    text = text.replace("peaches", "peach")
    text = text.replace("pears", "pear")
    text = text.replace("peppers", "pepper")
    text = text.replace("pickles", "pickle")
    text = text.replace("pies", "pie")
    text = text.replace("pizzas", "pizza")
    text = text.replace("plums", "plum")
    text = text.replace("popcorns", "popcorn")
    text = text.replace("porks", "pork")
    text = text.replace("potatoes", "potato")
    text = text.replace("poutines", "poutine")
    text = text.replace("pretzels", "pretzel")
    text = text.replace("puddings", "pudding")
    text = text.replace("pumpkins", "pumpkin")

    text = text.replace("raspberries", "raspberry")
    text = text.replace("rices", "rice")

    text = text.replace("salads", "salad")
    text = text.replace("salmons", "salmon")
    text = text.replace("sandwiches", "sandwich")
    text = text.replace("sausages", "sausage")
    text = text.replace("seaweeds", "seaweed")
    text = text.replace("shrimps", "shrimp")
    text = text.replace("smoothies", "smoothie")
    text = text.replace("sodas", "soda")
    text = text.replace("soups", "soup")
    text = text.replace("spaghettis", "spaghetti")
    text = text.replace("spinaches", "spinach")
    text = text.replace("squids", "squid")
    text = text.replace("strawberries", "strawberry")
    text = text.replace("sugars", "sugar")
    text = text.replace("sushis", "sushi")

    text = text.replace("teas", "tea")
    text = text.replace("tacos", "taco")
    text = text.replace("toasts", "toast")
    text = text.replace("tomatoes", "tomato")
    text = text.replace("tunas", "tuna")
    text = text.replace("turkeys", "turkey")

    text = text.replace("vegetables", "vegetable")

    text = text.replace("waters", "water")
    text = text.replace("watermelons", "watermelon")
    text = text.replace("wines", "wine")

    text = text.replace("yogurts", "yogurt")

    return text

def map_food(text):
     top_foods = ['apple', 'asparagus','avocado',
             'banana','bean','beef','bread','broccoli','burger', 'butter', 'blueberry', 'blackberry', 
             'candy', 'cake','caramel','carrot','caviar','cereal','cheese','chicken','chocolate', 'coffee', 'cookie', 'corn', 'crab','croissant','cucumber', 'curry', 
             'donut','duck', 
             'egg', 
             'fish','fry','fruit', 
             'garlic','ginger','goose','grape','grapefruit','guacamole', 
             'ham','honey','hummus', 
             'ice cream', 'iced', 
             'jelly','juice', 
             'kimchi','kiwi', 
             'lamb','lasagna','lemon','lentil','lettuce', 'lobster', 
             'mango','meat', 'matcha','milk','mint','mochi','muffin','mushroom', 
             'naan','noodle', 
             'oatmeal','octopus','olive','omelette','onion','orange', 
             'pancake', 'pasta','pastry','peach','pear','pepper','pickle','pie','pizza','plum', 'popcorn','pork','potato','poutine','pretzel','pudding','pumpkin', 
             'raspberry', 'rice', 
             'salad', 'salmon','sandwich','sausage','seaweed','shrimp','smoothie', 'soda','soup','spaghetti','spinach','squid','strawberry','sugar', 'sushi', 
             'tea', 'taco', 'toast','tomato','tuna','turkey', 
             'vegetable', 
             'water','watermelon', 'wine', 
             'yogurt']
     
     text = normalize(text)
     matches = []
     
     for food in top_foods:
        if re.search(r'\b' + re.escape(food) + r'\b', text):
            matches.append(food)

        return tuple(matches) if matches else text

def Q13_conversion(data: pd.DataFrame) -> None:
    data["Q13"] = (
        data["Q13"]
        .str.lower()
        .str.strip()
        .str.replace(r'[^a-z\s]', '', regex=True)   # remove punctuation
        .str.replace(r'\b(a|an|the)\b', '', regex=True)  # remove articles
        .str.replace(r'\s+', ' ', regex=True)  # clean extra spaces
    )

    data["Q13"] = data["Q13"].apply(map_food)

    mapped = data["Q13"].apply(map_food)

    food_counts = Counter()
    for entry in mapped:
        if isinstance(entry, tuple):
            food_counts.update(entry)
        else:
            food_counts.update([entry])

    popular_foods = {food for food, count in food_counts.items() if count >= 5}

    for food in popular_foods:
        col_name = f"food_{food.replace(' ', '_')}"
        
        data[col_name] = mapped.apply(lambda x: int(food in x) if isinstance(x, tuple) else int(food == x))

    indicator_cols = [col for col in data.columns if col.startswith("food_")]

    data = data[(data[indicator_cols].sum(axis=1) > 0)]

    data.drop("Q13", axis=1, inplace=True)

def create_bag_of_words_from_file(filename: str) -> Tuple[np.ndarray, list]:
    """
    Create binary bag of words from soundtrack and feeling description columns.
    
    Parameters:
        filename (str) : a string containing the path to a data set.

    Returns:
        X (np.ndarray) : Binary bag-of-words matrix with shape (n_responses, n_words)
        vocabulary (list) : List of all unique words found in the both columns
    """
    data = pd.read_csv(filename)
    
    soundtrack_col = 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.'
    feeling_col = 'Describe how this painting makes you feel.'
    
    all_texts = []
    for i in range(len(data)):
        combined = str(data[soundtrack_col][i]) + " " + str(data[feeling_col][i])
        all_texts.append(combined)

    doc_words = []
    all_words = []
    
    for text in all_texts:
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        
        doc_words.append(words)
        all_words.extend(words)
    
    vocabulary = sorted(set(all_words))
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    X = np.zeros((len(data), len(vocabulary)))
    
    for i, words in enumerate(doc_words):
        for word in set(words):  
            if word in word_to_idx:
                X[i, word_to_idx[word]] = 1
    
    return X, vocabulary

# Q2 and Q14 conversion
def textQ_conversion(data: pd.DataFrame) -> None:
    X_bow, vocabulary = create_bag_of_words_from_file("ml_challenge_dataset.csv")

    bow_columns = vocabulary  
    bow_df = pd.DataFrame(X_bow, columns=bow_columns)

    data = pd.concat([data, bow_df.loc[data.index]], axis=1)

    data.drop("Q2", axis=1, inplace=True)
    data.drop("Q14", axis=1, inplace=True)

def output_conversion(data: pd.DataFrame) -> None:
   # Create new mappings
    painting_mapping = {"The Persistence of Memory": 1, "The Water Lily Pond": 2, "The Starry Night": 3}

    # Map paintings to numerical values.
    data["Label"] = data["Label"].map(painting_mapping)

def normalizeTrainingData(data: pd.DataFrame, Features: list[str]) -> None:
    mean = data[Features].mean()
    std = data[Features].std()
    data[Features] = (data[Features] - mean) / std

def normalizeTestData(data: pd.DataFrame, Features: list[str], trainingMean, trainingStd) -> None:
    data[Features] = (data[Features] - trainingMean) / trainingStd

def createTrainingDataFrame(trainingDataFileName: str) -> pd.DataFrame:
    data = pd.read_csv(trainingDataFileName)
    rename_columns(data)
    question_conversions(data)
    Q13_conversion(data)
    textQ_conversion(data)
    output_conversion(data)
    normalizeTestData(data, ["Q7", "Q8", "Q9"])
    return data

def getTraining(trainFileName: str) -> tuple[np.ndarray, np.array]:
    data = createTrainingDataFrame(trainFileName)
    X_train = data.drop(columns=['unique_id', 'Label'])
    t_train = data["Label"]
    return X_train.to_numpy(), t_train.to_numpy()

