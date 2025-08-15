# 🪐 Exoplanet Habitability Predictor

This project predicts whether an exoplanet might be **habitable** based on its planet and star parameters.  
It uses a **machine learning model** trained on NASA exoplanet data and comes with an **interactive interface** using Gradio.

---

## 📂 Files in the project

- `EXO2.ipynb` — Notebook for data preprocessing, model training, and saving.  
- `exo_life_pipeline.pkl` — Saved machine learning pipeline (model + preprocessing).  
- `gradio_app.py` — Web interface to test planets interactively.  
- `requirements.txt` — List of Python packages needed to run the project.  

---

## 🚀 How it works

1. Enter planet and star properties (like size, mass, temperature).  
2. The model predicts **whether the planet is habitable**.  
3. You also get a **confidence score** showing how likely it is to be habitable.  

---

## 🛠 How to use

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/exoplanet-habitability.git
cd exoplanet-habitability


2. install dependencies
pip install -r requirements.txt

3. Run the gradio app
python gradio_app.py

4. Open the link shown in the terminal in your browser to test planets.

🌍 Example
Input example:
Orbital Period: 365 days
Semi-major Axis: 1.0 AU
Planet Radius: 1.0 Earth radii
Planet Mass: 1.0 Earth masses
Insolation: 1.0
Equilibrium Temp: 288 K
Star Temp: 5778 K
Star Radius: 1.0 Sun
Star Luminosity: 1.0 Sun

Output:
🌍 Habitable (Confidence: 0.79)
