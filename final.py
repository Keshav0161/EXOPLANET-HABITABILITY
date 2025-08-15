import gradio as gr
import joblib
import pandas as pd

# Load the saved model pipeline
pipeline = joblib.load("exo_life_pipeline.pkl")

def predict_habitability(pl_orbper, pl_orbsmax, pl_rade, pl_bmasse,
                         pl_insol, pl_eqt, st_teff, st_rad, st_lum, threshold=0.3):

    data = pd.DataFrame([[pl_orbper, pl_orbsmax, pl_rade, pl_bmasse,
                          pl_insol, pl_eqt, st_teff, st_rad, st_lum]],
                        columns=['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse',
                                 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_lum'])

    prob = pipeline.predict_proba(data)[:, 1][0]
    label = "🌍 Habitable" if prob >= threshold else "❌ Not Habitable"
    return f"{label} (Confidence: {prob:.2f})"

# Define UI
inputs = [
    gr.Number(label="Orbital Period (days)", value=365.0),
    gr.Number(label="Semi-major Axis (AU)", value=1.0),
    gr.Number(label="Planet Radius (Earth radii)", value=1.0),
    gr.Number(label="Planet Mass (Earth masses)", value=1.0),
    gr.Number(label="Insolation (Earth=1)", value=1.0),
    gr.Number(label="Equilibrium Temp (K)", value=288.0),
    gr.Number(label="Star Temp (K)", value=5778.0),
    gr.Number(label="Star Radius (Sun=1)", value=1.0),
    gr.Number(label="Star Luminosity (Sun=1)", value=1.0),
]

demo = gr.Interface(
    fn=predict_habitability,
    inputs=inputs,
    outputs="text",
    title="🪐 Exoplanet Habitability Predictor",
    description="Enter planet and star parameters to see if it might be habitable."
)

demo.launch(inline=True)  # Runs inside Jupyter
