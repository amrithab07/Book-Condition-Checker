import glob
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG, BS, SEED = 224, 16, 42

# ---- stratified split: 68% train / 12% val / 20% test ----
rows = []
for label, name in enumerate(['damaged_books', 'good_books']):
    for p in glob.glob(f'uploads/{name}/*'):
        if p.lower().endswith(('.jpg', '.jpeg', '.png')):
            rows.append((p, str(label)))
df = pd.DataFrame(rows, columns=['path', 'label'])
trainval, test_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
train_df, val_df = train_test_split(trainval, test_size=0.15, stratify=trainval.label, random_state=SEED)
print(len(train_df), len(val_df), len(test_df))

aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                         rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
                         zoom_range=0.2, brightness_range=(0.7, 1.3),
                         horizontal_flip=True, fill_mode='nearest')
plain = ImageDataGenerator(preprocessing_function=preprocess_input)

def gen(g, d, shuffle):
    return g.flow_from_dataframe(d, x_col='path', y_col='label', target_size=(IMG, IMG),
                                 batch_size=BS, class_mode='binary', shuffle=shuffle, seed=SEED)

train_g, val_g, test_g = gen(aug, train_df, True), gen(plain, val_df, False), gen(plain, test_df, False)

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df.label.astype(int))
class_weight = {0: cw[0], 1: cw[1]}

# ---- model ----
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG, IMG, 3))
base.trainable = False
inp = tf.keras.Input((IMG, IMG, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inp, out)

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# phase 1: train the head
model.compile(tf.keras.optimizers.Adam(1e-3), 'binary_crossentropy', metrics=['accuracy'])
model.fit(train_g, validation_data=val_g, epochs=10, class_weight=class_weight, callbacks=[early])

# phase 2: fine-tune the last 30 layers (BatchNorm stays frozen)
base.trainable = True
for l in base.layers[:-30]:
    l.trainable = False
for l in base.layers[-30:]:
    if isinstance(l, layers.BatchNormalization):
        l.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), 'binary_crossentropy', metrics=['accuracy'])
early2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model.fit(train_g, validation_data=val_g, epochs=25, class_weight=class_weight, callbacks=[early2])

model.save('book_condition_model_v2.h5')

# ---- predict helper (avoids the Windows hang) ----
def predict(g):
    probs, ys = [], []
    for i in range(len(g)):
        x, y = g[i]
        probs.extend(model.predict(x, verbose=0).ravel())
        ys.extend(y.astype(int))
    return np.array(probs), np.array(ys)

# ---- tune threshold on VALIDATION only ----
vp, vy = predict(val_g)
best_t = max(np.arange(0.3, 0.71, 0.05), key=lambda t: balanced_accuracy_score(vy, vp >= t))
print("Chosen threshold:", round(best_t, 2))

# ---- final test (touched once) ----
tp, ty = predict(test_g)
pred = (tp >= best_t).astype(int)
print(confusion_matrix(ty, pred))
print(classification_report(ty, pred, target_names=['damaged_books', 'good_books']))
print("Test accuracy:", np.mean(pred == ty))