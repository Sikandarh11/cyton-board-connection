from handler_combined import CombinedEEGHandler
path = "2. offline_pipeline\config_combined.json"

handler = CombinedEEGHandler(path)

print(handler.model_info())

# Single file
PATH = "Fyp-Transfer-Learning-model-Handlers-\\data\\ds_case.edf"
result = handler.predict_file(PATH, true_label="DS")
print("Final label :", result["final_label"])
print("Stage 1     :", result["stage1_prediction"], result["stage1_confidence"])
print("Stage 2     :", result["stage2_prediction"], result["stage2_confidence"])

# Save window-level detail
handler.save_result(result)

# Numpy array input
import numpy as np
eeg = np.random.randn(9, 25000).astype("float32")
result = handler.predict_array(eeg, subject_id="test_array")
print("Final label :", result["final_label"])

# Whole folder
df = handler.predict_folder("Fyp-Transfer-Learning-model-Handlers-\\data\\")
print(df[["subject_id", "final_label", "stage1_prediction",
          "stage2_prediction", "stage1_confidence"]])
'''

---

The decision logic in one place:
```
EDF / array
    │
    ▼
[Stage 1: DS vs Control]
    │
    ├── Control  ──→  "Control (Non-DS)"   ← pipeline stops
    │
    └── DS
         │
         ▼
    [Stage 2: DS vs Abnormal]
         │
         ├── DS        ──→  "DS"
         │
         └── Abnormal  ──→  "Abnormal (Non-DS)"


'''