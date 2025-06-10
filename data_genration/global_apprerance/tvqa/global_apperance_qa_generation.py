import os 
import re
import json 
import ast 
import random
# set the seed for reproducibility 
random.seed(72) # it is my birthday 7th of February
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description="Global appearance all shows")
parser.add_argument("--annotations_folder",required=False,default="generated_outfits_and_transitions")
parser.add_argument("--output_json",required=False,default="global_appearance_all_shows_with_grounding.json")

args = parser.parse_args()

global_appearance_MCQ_header="Choose the correct option for the following question:"
global_appearance_pool_of_questions=[
    "What is the correct sequence of changing the outfits for {} in this video?",
    "Can you outline the order of changing the outfits for {} in this video?",
    "What is the sequence of outfit changes for {} in this video?",
    "Can you list the order of outfits worn by {} in this video?",
    "What is the correct sequence of outfits {} wears in this video?",
    "How does {}'s clothing change from the beginning to the end of this video?",
    "In what order does {} change outfits in this video?",
    "What are the correct chronological outfit changes for {} in this video?",
    "How do {}'s clothes change throughout this video?",
    "What is the correct sequence of outfit changes for {} in this video?",
]

global_appearance_distractors = [
    ["blue jeans and white t-shirt", "black blazer and gray trousers", "navy blue gym shorts and tank top","red carpet gown with sequins"],
    ["pink pajamas", "white blouse and black pencil skirt", "red sequined party dress","red carpet gown with sequins"],
    ["khaki pants and blue polo shirt", "green soccer jersey and shorts", "black cocktail dress","black running shorts and a white tank top"],
    ["floral bathrobe", "yellow sundress", "white chef's coat and hat","black running shorts and a white tank top"],
    ["gray hoodie and ripped jeans", "silver evening gown", "purple loungewear set","floral spring dress","light blue summer dress"],
    ["red tracksuit", "blue school uniform with a plaid skirt", "gold sequin dress"],
    ["striped sleepwear set", "navy blue business suit", "light blue summer dress"],
    ["black yoga pants and a pink tank top", "red casual dress", "black tuxedo", "red casual dress", "black tuxedo"],
    ["green bikini", "denim shorts and a white blouse", "black evening gown"],
    ["white winter coat and brown boots", "floral spring dress", "red carpet gown with sequins"],
    ["black leggings and a gray sweatshirt", "blue jeans and a white sweater", "black tuxedo"],
    ["yellow raincoat", "gray business suit", "silver party dress","black running shorts and a white tank top"],
    ["blue flannel pajamas", "red weekend outfit with jeans and a t-shirt", "blue silk evening dress"],
    ["black yoga pants and purple sports bra", "white sundress", "black formal suit","black running shorts and a white tank top"],
    ["blue overalls and a white t-shirt", "red polka dot dress", "black cocktail dress"],
    ["gray sweatpants and a white hoodie", "navy blue blazer and khaki pants", "green evening gown"],
    ["black running shorts and a white tank top", "blue jeans and a red plaid shirt", "black suit and tie"],
    ["white apron and chef's hat", "denim jacket and black pants", "emerald green evening dress"],
    ["purple workout leggings and a pink top", "black business suit", "yellow summer dress","red swimsuit", "white t-shirt and khaki shorts", "blue evening gown"],
    ["red swimsuit", "purple workout leggings and a pink top","white t-shirt and khaki shorts", "blue evening gown"]
]

def global_appearance_generate_unique_options(correct_answer, num_options=5):
    global global_appearance_distractors
    # choose 4 random distractors without replacement
    distractor_1, distractor_2, distractor_3, distractor_4 = random.sample(global_appearance_distractors, 4)
    options = []
    all_events = correct_answer.copy()
    if len(all_events)==2:
        num_options=2
        options = [distractor_1, distractor_2, distractor_3]
        print("events are only 2", all_events)
    if len(all_events)==1:
        num_options=1
        options = [distractor_1, distractor_2, distractor_3,distractor_4]
        print("events are only 1", all_events)
    timer=0
    
    for _ in range(num_options):
        while True:
            timer+=1
            random.shuffle(all_events)
            option = all_events.copy()
            if option != correct_answer and option not in options:
                options.append(option)
                break
            if timer>100:
                break
            
    return options

 
benchmark_data={}  
annotations_folder=args.annotations_folder
n_questions=0
n_options=[]
for show in tqdm(os.listdir(annotations_folder)):
    for season in tqdm(os.listdir(os.path.join(annotations_folder,show))):
        for episode in tqdm(os.listdir(os.path.join(annotations_folder,show,season))):
            ann_path = os.path.join(annotations_folder,show,season,episode)
            global_apperance_data = json.load(open(ann_path, 'r'))
            episode=episode.split(".")[0]
            benchmark_data[f"/{show}/{season}/{episode}.mp4"]=[]
            for character in global_apperance_data:
                correct_answer = global_apperance_data[character]["outfits"].copy()
                grounding_times=global_apperance_data[character]["character_groundings"]
                options = global_appearance_generate_unique_options(correct_answer, num_options=4)
                options.append(correct_answer)
                # add I don't know option
                options.append("I don't know")
                random_q = random.choice(global_appearance_pool_of_questions)
                question = f"{global_appearance_MCQ_header} {random_q.format(character)}"
                # shuffle the options
                random.shuffle(options)
                if len(options) != 6:
                    print("number of options: ", len(options))
                
                answer_key = options.index(correct_answer)
                data = {}
                data['answer_idx'] = answer_key
                data['options'] = options
                data['question'] = question
                data['show'] = show
                data['season'] = season
                data['episode'] = episode
                data['source'] = "tvqa"
                data['character'] = character
                data['video_path_mp4'] = f"/{show}/{season}/{episode}.mp4"
                data['video_path_frames'] = f"/{show}/{season}/{episode}"
                data['video_subtitles'] = f"/{show}/{season}/{episode}.srt"
                data['temporal_grounding'] = grounding_times
                benchmark_data[f"/{show}/{season}/{episode}.mp4"].append(data)
                n_questions+=1
                n_options.append(len(options))
with open(args.output_json, 'w') as f:
    json.dump(benchmark_data, f, indent=4)
            
            
print("Number of questions: ", n_questions)
print("Max number of options: ", max(n_options))
print("Min number of options: ", min(n_options))
print("Average number of options: ", sum(n_options)/len(n_options))
print("Done generating global appearance questions for all shows")        

        
        
        
        