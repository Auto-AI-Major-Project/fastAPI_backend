import pickle
import os

# Path to your .pkl file in Downloads
pkl_path = os.path.expanduser("~/Downloads/model.pkl")

# Load the model
with open(pkl_path, "rb") as file:
    model = pickle.load(file)

# Print the model type
print("✅ Model loaded successfully!")
print("Model type:", type(model))

# If it’s a PyCaret model, check its internal estimator
try:
    print("\nUnderlying estimator:")
    print(model)
except Exception as e:
    print("Could not inspect model details:", e)

# Optional: check available attributes
print("\nModel attributes:")
print(dir(model))
