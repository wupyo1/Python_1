import tensorflow as tf
import numpy as np

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),  # Input Layer
    tf.keras.layers.Dense(units=2, activation='sigmoid'),  # Hidden Layer
    tf.keras.layers.Dense(units=1, activation='sigmoid')   # Output Layer
])

# 모델 컴파일
model.compile(loss='binary_crossentropy',  # 손실 함수
              optimizer='sgd',             # 확률적 경사 하강법(SGD)
              metrics=['accuracy'])        # 정확도 메트릭

# 입력 데이터와 출력 레이블
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # AND 게이트 입력
y = np.array([[0], [1], [1], [1]])              # OR 게이트 출력

# 모델 학습
model.fit(x, y, batch_size=1, epochs=7000, verbose=0)

# 모델 예측
predictions = model.predict(x)

# 결과 출력
print("Predictions:")
print(predictions)

#수정_0119
