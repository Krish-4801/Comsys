# taskB/train.py
import tensorflow as tf
import os
import sys

# Add project root to path to allow imports from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taskB.config import *
from taskB.utils import set_seed, save_training_history_plot
from taskB.data_loader import get_image_paths_by_person, create_image_pairs, create_tf_dataset
from taskB.model import get_embedding_model, get_siamese_model

def main():
    set_seed(SEED)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    except Exception as e:
        print(f"MirroredStrategy failed: {e}. Using default strategy.")
        strategy = tf.distribute.get_strategy()

    print("\n--- Preparing Training Data ---")
    person_data = get_image_paths_by_person(TRAIN_DATA_PATH)
    if not person_data: return

    anchors, comps, labels = create_image_pairs(person_data, MAX_TRAIN_SAMPLES)
    global_batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dataset = create_tf_dataset(anchors, comps, labels, global_batch_size, SEED)
    
    with strategy.scope():
        image_shape = IMG_SIZE + (3,)
        embedding_net = get_embedding_model(image_shape, EMBEDDING_DIM)
        siamese_net = get_siamese_model(embedding_net, image_shape)
        siamese_net.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            metrics=["accuracy"])
    siamese_net.summary()

    print("\n--- Starting Model Training ---")
    history = siamese_net.fit(train_dataset, epochs=EPOCHS, verbose=1)

    print("\n--- Training Complete ---")
    siamese_net.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    plot_path = os.path.join(RESULTS_DIR, "training_history.png")
    save_training_history_plot(history, plot_path)

if __name__ == "__main__":
    main()