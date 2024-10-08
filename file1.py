from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('data/pokemon_data.csv')

# Prepare the data
X = df.iloc[:, 5:7]  # Select 'Attack' and 'Defense'
Y = df.iloc[:, -1]   # Select 'Legendary'

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)  # Fit and transform on training data
x_test_scale = scaler.transform(x_test)  # Only transform on test data

# Initialize and fit the model
clf = LogisticRegression()
clf.fit(x_train_scale, y_train)  # Fit on scaled training data

# Make predictions
y_pred = clf.predict(x_test_scale)  # Use scaled test data for predictions

# Convert y_test and y_pred to integer for metrics calculation
y_test = np.array(y_test, dtype=int)
y_pred = np.array(y_pred, dtype=int)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

r_squared = r2_score(y_test, y_pred)
print("R-squared (R²):", r_squared)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot of Attack vs Defense
scatter = ax1.scatter(df['Attack'], df['Defense'], c=df['Legendary'].astype(int), cmap='viridis')
ax1.set_xlabel('Attack')
ax1.set_ylabel('Defense')
ax1.set_title('Attack vs Defense')
fig.colorbar(scatter, ax=ax1, label='Legendary (1: Yes, 0: No)')

# Plot decision regions
plot_decision_regions(X=x_train_scale, y=y_train.to_numpy().astype(np.int_), clf=clf, legend=2, ax=ax2)
ax2.set_xlabel('Attack (scaled)')
ax2.set_ylabel('Defense (scaled)')
ax2.set_title('Decision Regions for Logistic Regression')

# Show the combined plot
plt.tight_layout()
plt.show()
