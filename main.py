from utils import load_data
from src.models import get_models
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = load_data()
X = df.drop(columns='MEDV')
y = df['MEDV']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate
models = get_models()
for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)
