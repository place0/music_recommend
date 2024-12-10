import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications import ResNet50
from keras import layers, models
import tensorflow as tf

# 데이터 로드
excel_file = './popsong_800.xlsx'
df = pd.read_excel(excel_file)

# 이미지 경로 및 카테고리 정보 추출
image_folder = './images'
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
               filename.endswith('.jpg') or filename.endswith('.png')]
categories = df['category'].values  # 카테고리 정보


# 모델 생성 (예: ResNet50)
def create_resnet_model(input_shape=(224, 224, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    model = models.Model(inputs=base_model.input, outputs=x)
    return model


# 이미지 전처리 및 임베딩 추출 함수
def preprocess_image(image_path):
    # 이미지 전처리 예시 (리사이즈 등)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


# 이미지 임베딩 생성 함수
def extract_image_embeddings(image_paths):
    model = create_resnet_model()
    embeddings = []
    for image_path in image_paths:
        image = preprocess_image(image_path)  # 이미지 전처리 함수 (이미지 크기 맞추기 등)
        embedding = model.predict(image)
        embeddings.append(embedding.flatten())  # 플래튼하여 벡터화
    return np.array(embeddings)


# 이미지 임베딩 추출
image_embeddings = extract_image_embeddings(image_paths)


# 유사도 계산 함수
def calculate_similarity(image_embedding1, image_embedding2):
    return cosine_similarity([image_embedding1], [image_embedding2])[0][0]


# 카테고리별 유사도 계산 (카테고리가 같은 이미지들끼리 유사도 평균)
def calculate_category_similarity(image_embeddings, categories):
    category_similarities = {}

    # 카테고리별로 이미지 그룹화
    category_groups = {}
    for i, category in enumerate(categories):
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(i)

    # 카테고리 내 이미지들 간의 유사도 계산
    for category, indices in category_groups.items():
        category_similarities[category] = []

        # 같은 카테고리의 이미지들끼리 유사도 계산
        for i in indices:
            for j in indices:
                if i < j:  # 중복 계산 방지
                    similarity_score = calculate_similarity(image_embeddings[i], image_embeddings[j])
                    category_similarities[category].append(similarity_score)

    return category_similarities


# 다른 카테고리 간 유사도 계산 (예시로 카테고리 1과 3의 이미지 간 유사도 계산)
def calculate_cross_category_similarity(image_embeddings, categories, cat1, cat2):
    cat1_indices = [i for i, category in enumerate(categories) if category == cat1]
    cat2_indices = [i for i, category in enumerate(categories) if category == cat2]

    similarities = []
    for i in cat1_indices:
        for j in cat2_indices:
            similarity_score = calculate_similarity(image_embeddings[i], image_embeddings[j])
            similarities.append(similarity_score)

    return np.mean(similarities) if similarities else 0


# 카테고리별 유사도 계산
category_similarities = calculate_category_similarity(image_embeddings, categories)

# 결과 출력 (카테고리별 유사도 평균)
for category, similarities in category_similarities.items():
    print(f"\nCategory {category}의 평균 유사도: {np.mean(similarities):.4f}")

# 예시로 카테고리 1과 3 간의 유사도 평균 계산
cross_category_similarity = calculate_cross_category_similarity(image_embeddings, categories, 1, 3)
print(f"\n카테고리 1과 3 간의 평균 유사도: {cross_category_similarity:.4f}")
