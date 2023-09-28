import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import os
import random
import json

# Constante
SEED = 42
ROWS_PER_FRAME = 543
data_dir = "/kaggle/input/asl-signs"
landmark_fimes_dir = "/kaggle/input/asl-signs/train_landmark_files"


def seed_it_all(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_it_all()  # Reproducible

def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def read_json(path):
    with open(path, "r") as file:
        json_data = json.load(file)
    return json_data


path_train_df = pd.read_csv(data_dir + "/train.csv")
path_train_df["path"] = data_dir + "/" + path_train_df["path"]
display(path_train_df.head(2)), len(path_train_df)


s2p_map = read_json(os.path.join(data_dir, "sign_to_prediction_index_map.json"))
p2s_map = {v: k for k, v in s2p_map.items()}

encoder = lambda x: s2p_map.get(x)
decoder = lambda x: p2s_map.get(x)

path_train_df["label"] = path_train_df["sign"].map(encoder)
print(f"shape = {path_train_df.shape}")

path_train_df.head(2)


distribution_lenght = int(len(path_train_df) / 100)
frames = np.zeros(distribution_lenght)
for index, row in tqdm(path_train_df.iterrows(), total=distribution_lenght):
    if index > distribution_lenght - 1:
        break
    x = load_relevant_data_subset(row.path)
    frames[index] = x.shape[0]


print("minimum frames in sequence : ", frames.min())
print("maximum frames in sequence : ", frames.max())
print("mean of frames in the first 944 sequences", frames.mean())
print("median of frames in the first 944 sequences", np.median(frames))


plt.figure(figsize=(10, 5))
hist = plt.hist(frames, bins=100)

median = np.median(frames)
plt.plot(
    [median, median],
    [0, hist[0].max()],
    "--",
    c="red",
    linewidth=2,
    label=f"Median={median:.0f} frames",
)

plt.title("Distribution of number of frame per sequence over the dataset")
plt.xlabel("n-frames in one sequence")
plt.ylabel("Number of sequence with n-frames")

plt.legend()
plt.show()



def tf_nan_mean(x, axis=0):
    x_zero_insted_of_nan = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x_zero_for_nan_else_one = tf.where(
        tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)
    )
    x_out = tf.reduce_sum(x_zero_insted_of_nan, axis=axis) / tf.reduce_sum(
        x_zero_for_nan_else_one, axis=axis
    )
    return tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))


def tf_nan_std(x, axis=0):
    d = x - tf_nan_mean(x, axis=axis)
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis))


DROP_Z = False
print("Drop Z in data column :", DROP_Z, end="\n\n")

# Drop most of the face landmarks to reduce the dimensionality
LANDMARK = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348] + list(
    range(468, 543)
)

LENGHT_LANDMARK = len(LANDMARK)
N_DATA = len(["x", "y"]) if DROP_Z else len(["x", "y", "z"])
FIXED_FRAME = int(np.median(frames))
SHAPE = [FIXED_FRAME, LENGHT_LANDMARK, N_DATA]

print("Fixed Frame (shape[0]) =", FIXED_FRAME, end="\n\n")
print("Shape =", SHAPE, end="\n\n")

class FeatureGen(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        if x.shape[0] is None:  # for inference model
            n_frames = FIXED_FRAME
        else:
            n_frames = x.shape[0]

        # Drop "z" column
        if DROP_Z:
            x = x[:, :, 0:2]

        # NaN values become 0
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

        # Landmarks reduction
        # Select only the usefull landmark
        x = tf.gather(
            x,
            indices=LANDMARK,
            axis=1,
        )

        if FIXED_FRAME > n_frames:
            outputs = tf.image.resize(x, size=[SHAPE[0], SHAPE[1]], method="bilinear")
        else:
            outputs = tf.image.resize(x, size=[SHAPE[0], SHAPE[1]], method="nearest")

        return outputs


feature_converter = FeatureGen()

sample = load_relevant_data_subset(path_train_df.path[1])
prepocesse_sample = feature_converter(sample)
prepocesse_sample.shape, sample.shape


TOTAL_DATA_LENGHT = len(path_train_df)
DATA_LENGHT_EXPERIMENT = int(len(path_train_df) / 10)
print("Lenght of data for modeling :", DATA_LENGHT_EXPERIMENT)
print(f"Percentage of total data {DATA_LENGHT_EXPERIMENT/TOTAL_DATA_LENGHT*100:.1f}%")


def convert_row(row):
    x = load_relevant_data_subset(row.path)
    x = feature_converter(x)
    return x, row.label


def convert_and_save_data(data_lenght=DATA_LENGHT_EXPERIMENT):
    np_features = np.zeros([data_lenght] + SHAPE)
    np_labels = np.zeros(data_lenght)

    print(f"Total data to processe : {data_lenght}")
    print(f"Percentage of total data {data_lenght/TOTAL_DATA_LENGHT*100:.2f}%")
    for index, row in tqdm(path_train_df.iterrows(), total=data_lenght):
        if index > data_lenght - 1:
            break

        if index % (DATA_LENGHT_EXPERIMENT // 10) == 0:
            print(f"Data processed {index/data_lenght*100:.1f}%")

        data = load_relevant_data_subset(row.path)
        feature, label = convert_row(row)
        np_features[index] = feature
        np_labels[index] = label

    np.save("features.npy", np_features)
    np.save("labels.npy", np_labels)



try:
    features = np.load("/kaggle/working/features.npy")
    labels = np.load("/kaggle/working/labels.npy")
    print("Data Load successfully")
except:
    print("Loading DATA has fail... \nCreating DataSet")
    convert_and_save_data(DATA_LENGHT_EXPERIMENT)

    features = np.load("features.npy")
    labels = np.load("labels.npy")


X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=SEED
)

del (
    features,
    labels,
)  # delete, usefull with full data otherwise it fail (memorry issues)

buffer_size = int(DATA_LENGHT_EXPERIMENT / 10)

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.shuffle(buffer_size).batch(128).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_data = val_data.batch(128).prefetch(tf.data.AUTOTUNE)


quick_test_idx = np.random.randint(0, len(y_val), size=(10,))  # See after training
quick_test_X = np.take(X_val, quick_test_idx, axis=0)
quick_test_y = np.take(y_val, quick_test_idx, axis=0)
del X_train, X_val, y_train, y_val


def dense_block(units, drop):
    fc = layers.Dense(units)
    norm = layers.LayerNormalization()
    act = layers.Activation("relu")
    dropout = layers.Dropout(drop)
    return lambda x: dropout(act(norm(fc(x))))


def classifier(lstm_units, n_labels, drop):
    lstm = layers.ConvLSTM1D(
        filters=lstm_units, kernel_size=1
    )  # RNN capable of learning long-term dependencies.
    dropout = layers.Dropout(drop)
    flat = layers.Flatten()
    dense = layers.Dense(n_labels)
    outputs = layers.Activation("softmax", dtype="float32", name="predictions")
    return lambda x: outputs(dense(flat(dropout(lstm(x)))))


def get_model(
    encoder_units=[128, 64],
    drop=0.5,
    lstm_units=250,
    n_labels=250,
    shape=SHAPE,
    learning_rate=0.001,
):
    inputs = layers.Input(shape=shape)
    x = inputs

    for units in encoder_units:
        x = dense_block(units, drop)(x)

    outputs = classifier(lstm_units, n_labels, drop)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return model


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "./ASL_model",
            save_best_only=True,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
            verbose=False,
        ),
    ]


cb_list = get_callbacks()

model = get_model()
model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=cb_list
)

model = tf.keras.models.load_model("./ASL_model")
score = model.evaluate(val_data)

predictions = model.predict(quick_test_X, verbose=False).argmax(axis=1)

for true_label_id, pred_label_id in zip(quick_test_y, predictions):
    true_label = decoder(true_label_id)
    pred_label = decoder(pred_label_id)
    result = True if pred_label == true_label else False
    print(
        f"Prediction on test label : {pred_label.upper():<10} => True label {true_label.upper():<10} => {int(result)}"
    )

def plot_history(history, zoom=0):
    df = pd.DataFrame(history.history)
    n = len(df.columns)

    row = n // 2
    col = n // 2 + n % 2

    plt.figure(figsize=(5 * (col + 1) + zoom, 5 * row + zoom))
    for i, column in enumerate(df.columns):
        plt.subplot(row, col + 1, i + 1)
        plt.plot(df[f"{column}"], label=f"{column}")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel(f"{column}")
        plt.tight_layout(pad=2)  # padding

    plt.subplot(row, col + 1, n + 1)
    for column in df.columns:
        plt.plot(df[f"{column}"], label=f"{column}")
        plt.legend()
    plt.xlabel("epochs")

plot_history(history)


def get_inference_model(model):
    inputs = tf.keras.Input(shape=(543, 3), name="inputs")
    x = feature_converter(inputs)
    x = tf.expand_dims(x, axis=0)
    x = model(x)
    output = tf.keras.layers.Activation(activation="linear", name="outputs")(x)
    inference_model = tf.keras.Model(inputs=inputs, outputs=output)
    inference_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    return inference_model

inference_model = get_inference_model(model)
inference_model.summary()

demo_output = inference_model(load_relevant_data_subset(path_train_df.path[0]))
decoder(int(demo_output.numpy().argmax(axis=1)))


converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
"""
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
"""
tflite_model = converter.convert()

model_path = "model.tflite"
# Save the model.
with open(model_path, "wb") as f:
    f.write(tflite_model)
