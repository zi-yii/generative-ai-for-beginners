import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# model deployment
client=AzureOpenAI(
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version='2023-05-15'
)

model=os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT']

DATASET_NAME = '08-building-search-applications/embedding_index_3m.json'
SIMILARITY_THRESHOLD = 0.8


# load embedding index into pandas df
def load_dataset(source):
    pd_vectors = pd.read_json(source)
    return pd_vectors.drop(columns=["text"], errors="ignore").fillna("")


# cosine similarity fn
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# search embedding index for query
def get_videos(query, dataset, rows):
    video_vectors = dataset.copy()
    query_embeddings = client.embeddings.create(input=query, model=model).data[0].embedding
    
    # calculate similarity
    video_vectors['similarity'] = video_vectors['ada_v2'].apply(
        lambda x: cosine_similarity(np.array(query_embeddings), np.array(x))
    )

    # filter for videos with high similarity
    mask = video_vectors['similarity'] >= SIMILARITY_THRESHOLD
    video_vectors = video_vectors[mask].copy()

    # sort videos by similarity
    video_vectors = video_vectors.sort_values(by='similarity', ascending=False)

    # return top rows
    return video_vectors.head(rows)


# get results of search query
def display_results(videos, query):
    def _gen_yt_url(video_id, seconds):
        """convert time in format 00:00:00 to seconds"""
        return f"https://youtu.be/{video_id}?t={seconds}"
    
    print(f"\nVideos similar to '{query}':")
    for _, row in videos.iterrows():
        youtube_url = _gen_yt_url(row["videoId"], row["seconds"])
        print(f" - {row['title']}")
        print(f"   Summary: {' '.join(row['summary'].split()[:15])}...")
        print(f"   YouTube: {youtube_url}")
        print(f"   Similarity: {row['similarity']}")
        print(f"   Speakers: {row['speaker']}\n")



"""
Process
1. First, the Embedding Index is loaded into a Pandas Dataframe.
2. Next, the user is prompted to enter a query.
3. Then the `get_videos` function is called to search the Embedding Index for the query.
4. Finally, the `display_results` function is called to display the results to the user.
5. The user is then prompted to enter another query. This process continues until the user enters `exit`.
"""

pd_vectors = load_dataset(DATASET_NAME)
while True:
    query = input("Enter a query: ") # get user query from input
    if query == "exit":
        break
    videos = get_videos(query, pd_vectors, 5)
    display_results(videos, query)





