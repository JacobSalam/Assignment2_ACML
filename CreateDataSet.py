import pickle

PF = open("PulpFiction.txt", "r", encoding='utf8', errors='ignore')
IG = open("IngloriousBasterds.txt", "r", encoding='utf8', errors='ignore')
JB = open("JackieBrown.txt", "r", encoding='utf8', errors='ignore')
RD = open("ReservoirDogs.txt", "r", encoding='utf8', errors='ignore')
DJ = open("Django.txt", "r", encoding='utf8', errors='ignore')
scripts = [PF, IG, JB, RD, DJ]
dialogues = []

for movie in scripts:
    lines = movie.readlines()
    temp_dialogue = ""

    for line in lines:
        if line.strip().isupper():
            if len(temp_dialogue) > 0:
                dialogues.append(temp_dialogue)
                temp_dialogue = line.strip()
                continue
            else:
                temp_dialogue = line.strip()

        if len(temp_dialogue) > 0:
            temp_dialogue = temp_dialogue + " " + line.strip()

with open('all_d.list', 'wb') as d_file:
    pickle.dump(dialogues, d_file)

