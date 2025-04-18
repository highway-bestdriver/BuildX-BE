# app/task/train.py
from celery import shared_task
from app.utils.pubsub import publish_log
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

@shared_task(name="app.task.train.run_training")
def run_training(user_id, model_code, epochs, batch_size, learning_rate):
    print("✅ run_training task 들어옴!")
    print(f"🔥 학습 시작: user={user_id}")

    exec_globals = {}
    exec(model_code, exec_globals)

    model = exec_globals["model"]
    x_train = exec_globals["x_train"]
    y_train = exec_globals["y_train"]
    x_test = exec_globals["x_test"]
    y_test = exec_globals["y_test"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    for epoch in range(epochs):
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
        acc = round(float(history.history['accuracy'][0]) * 100, 2)
        loss = round(float(history.history['loss'][0]), 4)

        print(f"📡 Epoch {epoch+1} | acc={acc} loss={loss}")

        publish_log(f"user:{user_id}", {
            "type": "epoch_log",
            "epoch": epoch + 1,
            "accuracy": acc,
            "loss": loss,
        })

    # 최종 지표 계산
    y_pred_logits = model.predict(x_test)
    y_pred = np.argmax(y_pred_logits, axis=1)
    y_true = np.argmax(y_test, axis=1)

    precision = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
    recall = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
    f1 = round(f1_score(y_true, y_pred, average="macro") * 100, 2)

    loss, accuracy = model.evaluate(x_test, tf.keras.utils.to_categorical(y_true), verbose=0)
    loss = round(loss, 4)
    accuracy = round(accuracy * 100, 2)

    print(f"✅ 학습 완료! acc={accuracy}, loss={loss}")

    publish_log(f"user:{user_id}", {
        "type": "final_metrics",
        "accuracy": accuracy,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    publish_log(f"user:{user_id}", {
        "status": "학습 완료"
    })

    return {"message": "done"}