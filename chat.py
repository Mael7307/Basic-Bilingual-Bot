import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = ""
print("\n \n     ------------------------------")
print("     ||                          ||")
print("     ||  WELCOME TO THE AI CAFÈ  ||")
print("     ||                          ||")
print("     ------------------------------")
print("               ||       ||         ")
print("               ||       ||\n ")




print("My name is Kaitlyn, I speak English and French, let's chat! (type 'quit' to exit) ")
print("Mon nom est Kaitlyn, Je parle Anglais et Français, parle avec moi! (tapez 'quit' pour quitter) \n")

while True:
    language = input(f"{bot_name}: What language do you speak? Quel langue parler vous (eng/fr)?")
    lng = False
    eng = False
    fr = False

    if language in ["english", 'English', "eng"]:
        name = input(f"{bot_name}: what would you like me to call you ? ")
        lng = True
        eng = True
        break

    if language in ["francais","Francais",'fr']:
        name = input(f"{bot_name}: Comment voulez-vous que je vous appelle?")
        lng = False
        fr = True
        break

    if not lng:
        print(f"{bot_name}: I do not understand...je ne comprends pas!")

if eng:
    print(f"{bot_name}: Welcome to my AI cafè. I can tell you all about my shop, from items for sale, delivery times and payment options.")
    print(f"{bot_name}: I also love talking about pets and telling jokes!")

if fr:
    print(f"{bot_name} : Bienvenue dans mon café AI. Je peux vous parler de ma boutique, des articles à vendre, des temps de livraison et des options de paiement.")
    print(f"{bot_name} : J’aime aussi parler des animaux de compagnie et raconter des blagues!")

while True:
    sentence = input(name + ": ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    elif eng:
        print(f"{bot_name}: I do not understand...ask me another question")
    elif fr:
        print(f"{bot_name}: je ne comprends pas...demande moi une autre question")