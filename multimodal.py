import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications import ResNet50
import tensorflow as tf
import pickle
from sklearn.metrics import precision_score, recall_score, accuracy_score

# 데이터 로드
excel_file = './popsong_800.xlsx'
df = pd.read_excel(excel_file)

# 데이터프레임 확장 함수
def expand_dataframe(df, num_images_per_song=3):
    expanded_rows = []
    for _, row in df.iterrows():
        for _ in range(num_images_per_song):
            expanded_rows.append(row)
    expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    return expanded_df

# 데이터프레임 확장
df_expanded = expand_dataframe(df)

# BERT 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
text_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# ResNet50 모델 로드
def create_resnet_model(input_shape=(224, 224, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

image_model = create_resnet_model()

# 저장 및 로드 함수
def save_embeddings(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print(f"Embeddings loaded from {file_path}")
            return pickle.load(f)
    else:
        return None

# 텍스트 전처리 및 임베딩 생성
def preprocess_text(text):
    text = text.lower().strip()
    return text

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# 이미지 전처리 및 임베딩 생성
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

def extract_multiple_image_embeddings(image_paths):
    embeddings = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
        embedding = image_model.predict(image).flatten()
        embeddings.append(embedding)
    return np.mean(embeddings, axis=0)  # 평균 임베딩 생성

# 노래 제목에 따른 이미지 파일 이름 생성
def get_image_filenames_for_title(title, image_folder):
    filenames = []
    for filename in os.listdir(image_folder):
        if str(filename).startswith(str(title)) and filename.endswith(('.jpg', '.png')):
            filenames.append(os.path.join(image_folder, filename))
    return filenames

# 임베딩 크기 검증 함수
def validate_embeddings(df, text_embeddings, image_embeddings):
    if len(text_embeddings) != len(image_embeddings):
        raise ValueError("텍스트 및 이미지 임베딩의 개수가 일치하지 않습니다.")
    if len(df) != len(text_embeddings):
        raise ValueError("데이터프레임과 텍스트 임베딩의 개수가 일치하지 않습니다.")
    if len(df) != len(image_embeddings):
        raise ValueError("데이터프레임과 이미지 임베딩의 개수가 일치하지 않습니다.")

# 텍스트 및 이미지 임베딩 로드 또는 생성
text_embeddings_file = 'text_embeddings.pkl'
text_embeddings = load_embeddings(text_embeddings_file)
if text_embeddings is None:
    text_embeddings = []
    for _, row in df.iterrows():
        text_embeddings.append(get_bert_embedding(preprocess_text(row['lyrics'])).flatten())
    text_embeddings = np.array(text_embeddings)
    save_embeddings(text_embeddings_file, text_embeddings)

image_embeddings_file = 'image_embeddings.pkl'
image_embeddings = load_embeddings(image_embeddings_file)
if image_embeddings is None:
    image_folder = './images'
    image_embeddings = []
    for _, row in df.iterrows():
        title = row['title']
        image_files = get_image_filenames_for_title(title, image_folder)
        if image_files:
            image_embedding = extract_multiple_image_embeddings(image_files)
            image_embeddings.append(image_embedding)
        else:
            image_embeddings.append(np.zeros((128,)))  # 이미지가 없을 경우 0 벡터
    image_embeddings = np.array(image_embeddings)
    save_embeddings(image_embeddings_file, image_embeddings)

# 멀티모달 임베딩 생성
multimodal_embeddings_file = 'multimodal_embeddings.pkl'
multimodal_embeddings = load_embeddings(multimodal_embeddings_file)
if multimodal_embeddings is None:
    multimodal_embeddings = np.hstack((text_embeddings, image_embeddings))
    save_embeddings(multimodal_embeddings_file, multimodal_embeddings)

# 추천 시스템: 텍스트 입력 + 선택된 음악의 이미지
def recommend_song_with_text_and_image(user_input, selected_music_title, df, multimodal_embeddings, df_expanded):
    # 선택된 음악 제목이 데이터프레임에 있는지 확인
    if selected_music_title not in df['title'].values:
        raise ValueError(f"선택한 음악 제목 '{selected_music_title}'이 데이터베이스에 없습니다. 다시 입력해주세요.")

    # 원본 df에서 선택된 음악의 인덱스
    selected_index_in_df = df[df['title'] == selected_music_title].index[0]

    # df_expanded에서의 해당 인덱스를 찾기
    selected_index_in_expanded_df = df_expanded[df_expanded['title'] == selected_music_title].index[0]

    # 선택된 음악의 이미지 임베딩 생성
    image_folder = './images'
    image_files = get_image_filenames_for_title(selected_music_title, image_folder)
    if image_files:
        selected_image_embedding = extract_multiple_image_embeddings(image_files)
    else:
        selected_image_embedding = np.zeros((128,))  # 이미지가 없으면 0으로 처리

    # 사용자 입력 텍스트 임베딩 생성
    user_text_embedding = get_bert_embedding(preprocess_text(user_input)).flatten()

    # 사용자 멀티모달 임베딩 생성
    user_multimodal_embedding = np.hstack((user_text_embedding, selected_image_embedding))

    # 유사도 계산
    similarities = cosine_similarity([user_multimodal_embedding], multimodal_embeddings)[0]

    # 선택된 음악 제외
    similarities[selected_index_in_df] = -1  # 자기 자신 제거

    # 가장 유사한 음악 찾기
    most_similar_index = np.argmax(similarities)
    most_similar_title = df.iloc[most_similar_index]['title']
    most_similar_score = similarities[most_similar_index]

    return most_similar_title, most_similar_score

# 사용자 입력 처리
# 사용자 입력 처리
print("랜덤으로 선택된 음악 제목 중에서 하나를 선택하세요:")
random_choices = df.sample(5)
print(random_choices[['title']])
selected_music = input("선택한 음악 제목을 입력하세요: ")
user_input = input("새해 소원을 입력해주세요: ")

# 추천 결과
recommended_title, similarity_score = recommend_song_with_text_and_image(
    user_input, selected_music, df, multimodal_embeddings, df_expanded
)

print(f"추천된 음악 제목: {recommended_title} (유사도: {similarity_score:.4f})")


# 평가 함수 (Top-k 정확도 및 카테고리별 성능 분석 포함)
def evaluate_recommendation_system_top_k(df, multimodal_embeddings, k=5, num_samples=100):
    y_true = []
    y_pred_top_k = []

    category_wise_results = {category: {'correct': 0, 'total': 0} for category in df['category'].unique()}

    for _ in range(num_samples):
        # 랜덤으로 노래 제목 선택
        random_row = df.sample(1).iloc[0]
        selected_music_title = random_row['title']
        true_category = random_row['category']

        # 랜덤 텍스트 입력 생성 (간단히 노래 가사 일부를 사용)
        user_input = random_row['lyrics'][:50]

        # 추천 결과 (Top-k)
        selected_index_in_expanded_df = df_expanded[df_expanded['title'] == selected_music_title].index[0]

        # 원본 데이터프레임의 인덱스로 변환
        selected_index_in_original_df = selected_index_in_expanded_df // 3  # 3개 이미지마다 반복됨

        # 선택된 음악의 멀티모달 임베딩 생성
        user_text_embedding = get_bert_embedding(preprocess_text(user_input)).flatten()

        # 선택된 인덱스를 기반으로 multimodal_embeddings에서 데이터 가져오기
        selected_embedding = multimodal_embeddings[selected_index_in_original_df]

        # 텍스트 임베딩과 이미지 임베딩 결합
        user_multimodal_embedding = np.hstack((user_text_embedding, selected_embedding[-128:]))

        # 유사도 계산
        similarities = cosine_similarity([user_multimodal_embedding], multimodal_embeddings)[0]

        # 가장 유사한 k개 음악 찾기
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # 상위 k개 인덱스
        top_k_categories = df.iloc[top_k_indices]['category'].values

        # 실제 카테고리와 매칭
        y_true.append(true_category)
        y_pred_top_k.append(top_k_categories)

        # 카테고리별 성능 기록
        for category in top_k_categories:
            if category == true_category:
                category_wise_results[true_category]['correct'] += 1
            category_wise_results[true_category]['total'] += 1

    # Top-k 정확도 계산
    top_k_accuracy = sum(
        1 for true, preds in zip(y_true, y_pred_top_k) if true in preds
    ) / len(y_true)

    # 카테고리별 정확도 계산
    category_accuracy = {
        category: (data['correct'] / data['total'] if data['total'] > 0 else 0)
        for category, data in category_wise_results.items()
    }

    print(f"Top-{k} Accuracy: {top_k_accuracy:.4f}")
    print("Category-wise Accuracy:")
    for category, acc in category_accuracy.items():
        print(f"  {category}: {acc:.4f}")

    return top_k_accuracy, category_accuracy


# 시스템 평가 실행
evaluate_recommendation_system_top_k(df_expanded, multimodal_embeddings, k=5, num_samples=100)
