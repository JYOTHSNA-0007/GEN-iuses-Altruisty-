# chat_bot.py

import torch
import torch.nn as nn
import joblib

MODEL_PATH = "model/travel_model.pt"
VECTORIZER_PATH = "model/vectorizer.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

# Load encoders
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Define model
class TravelModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)

model = TravelModel(input_dim, output_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("🧠 AI TravelBot is ready! Type your travel request (or type 'exit'):")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Goodbye!")
        break

    X = vectorizer.transform([user_input]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)
        predicted = torch.argmax(output, dim=1)
        attraction = label_encoder.inverse_transform(predicted.numpy())[0]
        print(f"\n🤖 TravelBot recommends: {attraction}")
