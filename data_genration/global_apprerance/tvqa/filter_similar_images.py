import os 
from PIL import Image
import torch 
import torchvision.transforms as T
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse 



dino_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").to(dino_device)
transform_image_pipeline = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def get_image_embeddings(pil_images):
    transformend_images = torch.stack([transform_image_pipeline(img)[:3]for img in pil_images]).to(dino_device)
    with torch.no_grad():
        embeddings = dinov2_vitb14(transformend_images)
    embeddings = embeddings.cpu().numpy()
    # embeddings already normalized
    return embeddings


argparser=argparse.ArgumentParser()
argparser.add_argument("--show",type=str,default="friends")
argparser.add_argument("--season",type=str,default="season_1")
argparser.add_argument("--characerts_path",type=str,default="characters")
argparser.add_argument("--output_path",type=str,default="characters_filtered_remove_similar_dino")
args=argparser.parse_args()
unfiltered_characerts_path=args.characerts_path
save_dir= args.output_path
os.makedirs(save_dir,exist_ok=True)

show=args.show
season=args.season

for episode in tqdm(os.listdir(os.path.join(unfiltered_characerts_path,show,season)),desc="Episodes"):
    for character in os.listdir(os.path.join(unfiltered_characerts_path,show,season,episode)):
        character_images_path=os.path.join(unfiltered_characerts_path,show,season,episode,character)
        save_character_path=os.path.join(save_dir,show,season,episode,character)
        os.makedirs(save_character_path,exist_ok=True)
        pil_images=[]
        image_names=[]
        for image_name in sorted(os.listdir(character_images_path)):
            image_names.append(image_name)
            image_path=os.path.join(character_images_path,image_name)
            image = Image.open(image_path)
            pil_images.append(image)
        frames_embeddings = get_image_embeddings(pil_images)
        similarities = []
        for i in range(len(frames_embeddings)-1):
            cosine_similarity = np.dot(frames_embeddings[i], frames_embeddings[i+1]) / (np.linalg.norm(frames_embeddings[i]) * np.linalg.norm(frames_embeddings[i+1]))
            cosine_similarity = np.round(cosine_similarity, 2)
            similarities.append(cosine_similarity)
        indecies_to_remove=set()
        for i in range(len(similarities)):
            if similarities[i]>0.60 :
                indecies_to_remove.add(i+1)
        indecies_to_remove=list(indecies_to_remove)
        # print(indecies_to_remove)
        indecies_to_remove.sort(reverse=True)
        for i in range(len(similarities)):
            if i not in indecies_to_remove:
                image_path=os.path.join(character_images_path,image_names[i])
                save_image_path=os.path.join(save_character_path,image_names[i])
                os.system(f"cp '{image_path}' '{save_image_path}'")
