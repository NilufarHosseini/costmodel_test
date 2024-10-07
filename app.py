import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset to simulate project costs
np.random.seed(42)
n_samples = 1000
project_type = np.random.choice([0, 1], size=n_samples)  # 0: Small, 1: Large project
project_lifetime = np.random.randint(6, 36, size=n_samples)  # Project lifetime in months
planned_cost = np.random.randint(50000, 500000, size=n_samples)  # Planned project cost
progress_percentage = np.random.choice([i for i in range(10, 110, 10)], size=n_samples)  # Progress in %
number_of_team_members = np.random.randint(5, 50, size=n_samples)  # Number of team members
project_complexity = np.random.randint(1, 10, size=n_samples)  # Complexity (1 to 10 scale)
actual_cost = planned_cost * (0.9 + (0.2 * np.random.rand(n_samples)))  # Actual cost

# Combine into a DataFrame
df = pd.DataFrame({
    'project_type': project_type,
    'project_lifetime': project_lifetime,
    'planned_cost': planned_cost,
    'progress_percentage': progress_percentage,
    'number_of_team_members': number_of_team_members,
    'project_complexity': project_complexity,
    'actual_cost': actual_cost
})

# Add interaction features (calculated based on other features)
df['cost_per_team_member'] = df['planned_cost'] / df['number_of_team_members']
df['lifetime_per_team_member'] = df['project_lifetime'] / df['number_of_team_members']
df['progress_weighted_cost'] = df['planned_cost'] * (df['progress_percentage'] / 100)

# Split the data into features and target
X = df.drop(columns='actual_cost')
y = df['actual_cost']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Prediction function for Gradio
def predict_project_costs(project_type, project_lifetime, planned_cost, number_of_team_members, project_complexity):
    # Create an empty list to store the predictions
    predictions = []

    # Loop through progress percentages from 10% to 100% in steps of 10%
    for progress in range(10, 110, 10):
        # Prepare the input data for the current progress
        input_data = pd.DataFrame({
            'project_type': [project_type],
            'project_lifetime': [project_lifetime],
            'planned_cost': [planned_cost],
            'progress_percentage': [progress],
            'number_of_team_members': [number_of_team_members],
            'project_complexity': [project_complexity]
        })
        
        # Calculate the interaction features based on the input data
        input_data['cost_per_team_member'] = input_data['planned_cost'] / input_data['number_of_team_members']
        input_data['lifetime_per_team_member'] = input_data['project_lifetime'] / input_data['number_of_team_members']
        input_data['progress_weighted_cost'] = input_data['planned_cost'] * (input_data['progress_percentage'] / 100)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict the project cost using the trained model
        predicted_cost = model.predict(input_data_scaled)[0]
        
        # Append the progress and predicted cost to the predictions list
        predictions.append((progress, round(predicted_cost, 2)))

    # Convert predictions to a pandas DataFrame for display
    results_df = pd.DataFrame(predictions, columns=["Project Progress (%)", "Predicted Cost"])
    return results_df

# Set up the Gradio interface
gr_interface = gr.Interface(
    fn=predict_project_costs,
    inputs=[
        gr.components.Dropdown(choices=[0, 1], label="Project Type (0: Small, 1: Large)"),
        gr.components.Slider(6, 36, step=1, label="Project Lifetime (in months)"),
        gr.components.Number(label="Total Planned Cost"),
        gr.components.Slider(5, 50, step=1, label="Number of Team Members"),
        gr.components.Slider(1, 10, step=1, label="Project Complexity (1 to 10)")
    ],
    outputs=gr.components.Dataframe(label="Predicted Costs at Different Project Progress Levels"),
    title="Project Cost Predictor with Interaction Features",
    description="Enter the project details, and the model will predict the project cost at each progress level (from 10% to 100%), including interaction features."
)

# Launch the Gradio web app
if __name__ == "__main__":
    gr_interface.launch(server_name="0.0.0.0", server_port=8080)
