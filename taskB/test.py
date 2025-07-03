# taskB/test.py
import tensorflow as tf
import numpy as np
import os
import argparse
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taskB.config import *
from taskB.data_loader import get_image_paths_by_person, create_image_pairs, create_tf_dataset
from taskB.model import CUSTOM_OBJECTS

def main(args):
    if not os.path.exists(args.data_path):
        print(f"Error: Test data path not found: '{args.data_path}'")
        return
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: '{args.model_path}'")
        return

    print(f"\n--- Loading model from {args.model_path} ---")
    model = tf.keras.models.load_model(args.model_path, custom_objects=CUSTOM_OBJECTS)
    model.summary()

    print("\n--- Preparing Test Data ---")
    person_data = get_image_paths_by_person(args.data_path)
    if not person_data: return

    anchors, comps, true_labels = create_image_pairs(person_data, args.max_samples)
    if len(true_labels) == 0:
        print("Could not generate test pairs from the data.")
        return

    test_dataset = create_tf_dataset(anchors, comps, true_labels, args.batch_size, SEED, shuffle=False)
    
    print("\n--- Running Inference ---")
    predictions = model.predict(test_dataset, verbose=1)
    pred_labels = (predictions.flatten() > 0.5).astype(int)

    print("\n--- Evaluation Metrics ---")
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔄 Recall:    {recall:.4f}")
    print(f"📈 F1-Score:  {f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(true_labels, pred_labels, target_names=['Non-Match (0)', 'Match (1)']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Siamese Network model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the test data directory.")
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH, help=f"Path to the model file. Default: {MODEL_SAVE_PATH}")
    parser.add_argument('--max_samples', type=int, default=MAX_TEST_SAMPLES, help=f"Max test pairs. Default: {MAX_TEST_SAMPLES}")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for prediction. Default: 128")
    
    args = parser.parse_args()
    main(args)