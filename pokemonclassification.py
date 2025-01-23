import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("rounakbanik/pokemon")

# Construct the file path
home_dir = os.path.expanduser('~')
file_path = os.path.join(home_dir, '.cache', 'kagglehub', 'datasets', 'rounakbanik', 'pokemon', 'versions', '1', 'pokemon.csv')

# Load the dataset
df = pd.read_csv(file_path)

# Preprocess the data
# Encode the target 'type1'
label_encoder = LabelEncoder()
df['type1'] = label_encoder.fit_transform(df['type1'])

# Select features and target
features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
            'against_bug', 'against_dark', 'against_dragon', 'against_electric',
            'against_fairy', 'against_fight', 'against_fire', 'against_flying',
            'against_ghost', 'against_grass', 'against_ground', 'against_ice',
            'against_normal', 'against_poison', 'against_psychic', 'against_rock',
            'against_steel', 'against_water']

X = df[features]
y = df['type1']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a custom dataset
class PokemonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PokemonDataset(X_train, y_train)
test_dataset = PokemonDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class PokemonClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PokemonClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = PokemonClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')