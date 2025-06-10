import os
from openai import OpenAI
import ast
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image


# GPT4O model
api_key=os.getenv("OPENAI_API_KEY")
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_unique_outfit(images_paths):
    # Getting the base64 string
    content = [
        {
            "type": "text",
            "text": "Given this sequence of images, identify transitions between different outfits for the character in the images by providing the unique outfits and the indices of the images where the transitions occur."
        },
        
    ]
    for i,image_path in enumerate(images_paths):
        base64_image=encode_image(image_path) 
        content.append(
            {
            "type": "text",
            "text": f"Image index: {i}",
            },
        )
        content.append(
        {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
        },
        }
        )
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an intelligent vision system that can track changes in the outfit of a character in a sequence of images. Identify transitions between different outfits for the character in the images.Your output should strictly follow this JSON format: {\"outfits\": \"<outfit_transitions>\",\"indices\": \"<transitions_indices>\"}, where '<outfit_transitions>' is a Python list containing the DETAILED descriptions of the unique outfits in the sequence of images and '<transitions_indices>' is a Python list containing the indices of the images where the transitions occur. the descriptions of the outfits should be detailed and should not contain any other information."
                    }
                ]
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 4096,
        "response_format":{ "type": "json_object" }
    }
    return payload

def send_requests_to_gpt4o(payload,index):
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        response_json=ast.literal_eval(response.json()['choices'][0]['message']['content'])
    except:
        print("Error in response: ",response.json())
        response_json=None
    return index,response_json

def get_all_character_appearance_intervals(seconds_array):
    intervals=[]
    start=seconds_array[0]
    for i in range(len(seconds_array)-1):
        if seconds_array[i+1]-seconds_array[i]>5:
            intervals.append((start,seconds_array[i]))
            start=seconds_array[i+1]
    intervals.append((start,seconds_array[-1]))
    return intervals

def get_characters_groundings(all_intervals, transition_seconds):
    """
    Assigns grounding intervals to each transition second such that 
    each interval can only be assigned to one transition second.

    Args:
        all_intervals (list of tuple): A list of intervals (start, end).
        transition_seconds (list of int): A list of transition seconds.

    Returns:
        list of list of tuple: A list where each entry corresponds to a transition second 
                               and contains the intervals assigned to that second.
    """
    groundings = []
    used_intervals = set()  # To track intervals that are already assigned

    for transition in transition_seconds[1:]:
        current_grounding = []

        for interval in all_intervals:
            if interval not in used_intervals and interval[1] <= transition:
                current_grounding.append(interval)
                used_intervals.add(interval)

        groundings.append(current_grounding)
    # Extend the last grounding to the end of the episode
    groundings.append([interval for interval in all_intervals if interval not in used_intervals])
    assert len(groundings)==len(transition_seconds)

    return groundings

def resize_images(images_paths):
    for image_path in images_paths:
        img=Image.open(image_path)
        img=img.resize((512,512))
        img.save(image_path)
        
argparser=argparse.ArgumentParser()
argparser.add_argument("--show",type=str,default="friends")
argparser.add_argument("--season",type=str,default="season_1")
argparser.add_argument("--filtered_images_folder",type=str,default="characters_filtered_remove_similar_dino")
argparser.add_argument("output_path",type=str,default="generated_outfits_and_transitions")
argparser.add_argument("original_images_dir",type=str,default="characters",help="Path to the original detected characters images before filtering")
args=argparser.parse_args()   
filtered_characerts_dir=args.filtered_images_folder
save_dir= args.output_path
os.makedirs(save_dir,exist_ok=True)
original_images_dir="characters" 


show=args.show
season=args.season
mian_characters_path="tvqa_main_characters.json"
with open(mian_characters_path) as f:
    main_characters=json.load(f)

total_number_of_images=0
Number_of_questions=0
m_character=set()
number_of_skipped_episodes=0
for show in os.listdir(filtered_characerts_dir):
    # if show=='bbt':
    #     continue
    for season in os.listdir(os.path.join(filtered_characerts_dir,show)):
        main_characters_list=main_characters[show]
        for episode in tqdm(os.listdir(os.path.join(filtered_characerts_dir,show,season)),desc="Episodes"):
            # if episode!="episode_1":
            #     continue
            epiosde_data={}
            save_json_path=os.path.join(save_dir,show,season)
            os.makedirs(save_json_path,exist_ok=True)
            exceptions=False
            for character in os.listdir(os.path.join(filtered_characerts_dir,show,season,episode)):
                character_images_path=os.path.join(filtered_characerts_dir,show,season,episode,character)
                images_name=sorted(os.listdir(character_images_path))
                if len (images_name)==0: # No images for this character
                    continue
                original_images_path=os.path.join(original_images_dir,show,season,episode,character)
                if len (original_images_path)<40 or character not in main_characters_list: # not a main character in this episode
                    continue
                m_character.add(character)
                original_images_names=sorted(os.listdir(original_images_path))
                seconds_array=[int(image_name.split(".")[0]) for image_name in original_images_names]
                # print(seconds_array)
                intervals=get_all_character_appearance_intervals(seconds_array)
                # print(intervals)
                images_paths=[os.path.join(character_images_path,image_name) for image_name in images_name]
                # resize_images(images_paths)
                # print("Number of images: ",len(images_paths))
                total_number_of_images+=len(images_paths)
                Number_of_questions+=1
                payload=get_unique_outfit(images_paths)
                if os.path.exists(os.path.join(save_json_path,f"{episode}.json")):
                    print("Reading the result from the saved json file",os.path.join(save_json_path,f"{episode}.json"))
                    with open(os.path.join(save_json_path,f"{episode}.json"),'r') as f:
                        result_data=json.load(f)
                    if character not in result_data:
                        continue
                    result=result_data[character]
                else: 
                    print("Sending request to GPT4O")
                    index,result=send_requests_to_gpt4o(payload,0)
                if result is None:
                    print("Error in the result, skipping this episode")
                    exceptions=True
                    number_of_skipped_episodes+=1
                    break
                if isinstance(result['indices'],str):
                    result['indices']=ast.literal_eval(result['indices'])
                indices=[int(res) for res in result['indices']]
                outfits=result['outfits']
                if len(outfits)-len(indices)==1:
                    if 0 not in indices:
                        indices.insert(0,0)
                if len(outfits)!=len(indices):
                    print("Error in the number of outfits and indices")
                    continue
                print("Character: ",character)
                print("len indices: ",len(indices))
                print("len outfits: ",len(outfits))
                print("len images: ",len(images_name))
                transition_images=[]
                transition_seconds=[]
                for idx in indices:
                    transition_images.append(images_name[idx])
                    transition_seconds.append(int(images_name[idx].split(".")[0]))
                # print(indices_seconds)
                character_groundings=get_characters_groundings(intervals,transition_seconds)
                epiosde_data[character]={
                    "outfits":outfits,
                    "transition_seconds":transition_seconds,
                    "indices":indices,
                    "transition_images":transition_images,
                    "character_groundings":character_groundings,
                    'all_characters_appearance':intervals,
                    
                }
            if exceptions:
                continue
            with open(os.path.join(save_json_path,f"{episode}.json"),"w") as f:
                json.dump(epiosde_data,f)
            print(f"Saved {episode}.json")
                    
print("Total number of images: ",total_number_of_images)
print("Total number of questions: ",Number_of_questions)
print("Main characters: ",m_character)
print("Number of main characters: ",len(m_character))
print("Number of skipped episodes: ",number_of_skipped_episodes)
