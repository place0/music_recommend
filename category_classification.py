import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications import ResNet50
from keras import layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 로드
excel_file = './popsong_800.xlsx'
df = pd.read_excel(excel_file)

# 이미지 경로 및 카테고리 정보 추출
image_folder = './images'
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
               filename.endswith('.jpg') or filename.endswith('.png')]
categories = df['category'].values  # 카테고리 정보
categories = np.repeat(df['category'].values, 3)

# 카테고리 라벨 인코딩
label_encoder = LabelEncoder()
categories_encoded = label_encoder.fit_transform(categories)

# 모델 생성 (ResNet50)
def create_resnet_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)  # Dense layer for better representation
    model = models.Model(inputs=base_model.input, outputs=x)  # 마지막 classification layer는 제외하고, 임베딩 벡터만 추출
    return model

# 이미지 전처리 함수
def preprocess_image(image_path):
    # 이미지 전처리 예시 (리사이즈 등)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# 이미지 임베딩 생성 함수 (이미지 데이터를 모델에 입력하여 임베딩을 추출)
def extract_image_embeddings(image_paths):
    model = create_resnet_model(input_shape=(224, 224, 3), num_classes=5)
    embeddings = []
    for image_path in image_paths:
        image = preprocess_image(image_path)  # 이미지 전처리
        embedding = model.predict(image)  # 모델을 사용해 이미지 임베딩 추출
        embeddings.append(embedding.flatten())  # 플래튼하여 벡터화
    return np.array(embeddings)

# 이미지 임베딩 추출
# 이미지 임베딩 추출 (임베딩 계산 후 저장)
def save_image_embeddings(image_paths, embeddings_file='image_embeddings.npy'):
    if os.path.exists(embeddings_file):
        print("이미지 임베딩 파일이 이미 존재합니다. 파일을 로드합니다.")
        embeddings = np.load(embeddings_file)
    else:
        embeddings = extract_image_embeddings(image_paths)
        np.save(embeddings_file, embeddings)
        print("이미지 임베딩을 저장했습니다.")
    return embeddings

# 이미지 임베딩을 저장하고 로드
image_embeddings = save_image_embeddings(image_paths)

# 데이터를 훈련, 검증, 테스트로 분할
X_train, X_temp, y_train, y_temp = train_test_split(image_embeddings, categories_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 분류 모델 학습
def create_classification_model(input_shape=(128,), num_classes=5):
    model = models.Sequential()

    # 입력 임베딩 벡터를 받는 레이어
    model.add(layers.InputLayer(input_shape=input_shape))

    # 첫 번째 Dense 레이어 - 임베딩 벡터에서 중요한 특성 추출
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())  # 배치 정규화 추가
    model.add(layers.Dropout(0.3))  # 과적합 방지를 위한 드롭아웃

    # 두 번째 Dense 레이어 - 특성 공간을 확장하여 더 많은 학습
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # 세 번째 Dense 레이어 - 카테고리 예측을 위한 최종 레이어
    model.add(layers.Dense(num_classes, activation='softmax'))  # 카테고리 예측을 위한 소프트맥스 출력

    return model

# 모델 훈련
classification_model = create_classification_model(input_shape=(image_embeddings.shape[1],), num_classes=len(np.unique(categories)))
classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = classification_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 모델 평가
test_loss, test_acc = classification_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
