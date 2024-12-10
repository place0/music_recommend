import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import pickle

# 데이터 구축
excel_file = './popsong_800.xlsx'
df = pd.read_excel(excel_file)

# BERT 모델 구축
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# 영어와 한국어 불용어
nltk.download('stopwords')
english_stopwords = stopwords.words('english')
korean_stopwords = ['을', '를', '이', '가', '에', '의', '에서', '으로', '도', '들', '은', '는', '과', '와', '한', '하다', '하', '때', '로', '보다', '이', '것', '저', '그', '있다', '없다', '같다']
stop_words = english_stopwords + korean_stopwords

# 텍스트 -> BERT 임베딩 변환 함수
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# keyword 정의
keywords = ["사랑", "돈", "행복", "성공", "욕심쟁이"]

# text 전처리 및 불용어 제거
def preprocess_text(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# 사용자 입력 text 가장 유사한 keyword 찾는 함수
def find_best_keyword(input_text):
    input_text = preprocess_text(input_text)
    input_keyword_similarities = {keyword: calculate_similarity(input_text, keyword) for keyword in keywords}
    input_best_keyword = max(input_keyword_similarities, key=input_keyword_similarities.get)
    return input_best_keyword

# BERT embedding 미리 계산하고 캐시하기
def get_all_song_embeddings(df):
    song_embeddings = {}
    for index, row in df.iterrows():
        song_title = row['title']
        song_lyrics = row['lyrics']
        song_lyrics = preprocess_text(song_lyrics)  # 불용어 제거
        song_embeddings[song_title] = get_bert_embedding(song_lyrics)
    return song_embeddings

# similarity 계산
def calculate_similarity(text1, text2):
    vec1 = get_bert_embedding(text1)
    vec2 = get_bert_embedding(text2)
    return cosine_similarity(vec1, vec2)[0][0]

# lyrics에 대한 similarity 기반으로 가장 유사한 노래를 찾는 함수
def find_most_similar_song(input_text, df, song_embeddings):
    best_similarity = -1
    best_title = ""
    best_keyword = ""

    input_best_keyword = find_best_keyword(input_text)

    print(f"입력된 텍스트는 '{input_best_keyword}'와 가장 유사합니다.")

    # 모든 노래에 대해 keyword에 대한 similarity 계산
    for index, row in df.iterrows():
        song_title = row['title']
        song_lyrics = row['lyrics']
        song_lyrics = preprocess_text(song_lyrics)

        song_embedding = song_embeddings[song_title]

        similarity = calculate_similarity(input_text, song_lyrics)

        if similarity > best_similarity:
            best_similarity = similarity
            best_title = song_title
            best_keyword = input_best_keyword

    return best_title, best_similarity, best_keyword

# 미리 계산된 embedding 불러오기
if os.path.exists('song_embeddings.pkl'):
    with open('song_embeddings.pkl', 'rb') as f:
        song_embeddings = pickle.load(f)
else:
    song_embeddings = get_all_song_embeddings(df)
    with open('song_embeddings.pkl', 'wb') as f:
        pickle.dump(song_embeddings, f)

# 사용자 입력을 받은 후, similarity 높은 곡과 similarity score 출력
input_text = input("새해 소원을 입력해주세요: ")
most_similar_song, similarity_score, keyword = find_most_similar_song(input_text, df, song_embeddings)
print(f"새해에 듣기 좋은 음악은?? : {most_similar_song} [유사도 점수 : {similarity_score}] [유사한 키워드 : {keyword}]")
