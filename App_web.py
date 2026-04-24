import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Predictor de Procrastinación", page_icon="🧠")
st.title("🧠 Predictor Inteligente de Procrastinación")

# ----------------------------
# GUARDAR DATOS
# ----------------------------
def guardar_datos(nuevo_dato):
    archivo = "datos_usuarios.csv"

    if os.path.exists(archivo) and os.path.getsize(archivo) > 0:
        try:
            df = pd.read_csv(archivo)
        except:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([nuevo_dato])], ignore_index=True)
    df.to_csv(archivo, index=False)

# ----------------------------
# DATOS
# ----------------------------
data_inicial = pd.DataFrame({
    "sueno": [4,6,7,5,8,3,6,7,5,4],
    "redes": [120,60,30,90,20,150,80,40,100,110],
    "energia": [2,3,4,2,5,1,3,4,2,2],
    "dificultad": [5,3,2,4,1,5,3,2,4,5],
    "procrastino": [1,0,0,1,0,1,1,0,1,1]
})

if os.path.exists("datos_usuarios.csv") and os.path.getsize("datos_usuarios.csv") > 0:
    data_usuario = pd.read_csv("datos_usuarios.csv")
    data = pd.concat([data_inicial, data_usuario], ignore_index=True)
else:
    data = data_inicial

# 🔥 transformación redes
data["redes_log"] = np.log1p(data["redes"])

# ----------------------------
# MODELO
# ----------------------------
X = data[["sueno", "redes_log", "energia", "dificultad"]]
y = data["procrastino"]

if len(y.unique()) > 1:
    modelo = LogisticRegression(C=0.5)
    modelo.fit(X, y)
else:
    modelo = None

# ----------------------------
# INPUTS
# ----------------------------
st.markdown("## 🎯 Ingresa tus hábitos")

col1, col2 = st.columns(2)

with col1:
    sueno = st.slider("😴 Horas de sueño", 0, 10, 6)
    energia = st.slider("⚡ Energía (1-5)", 1, 5, 3)

with col2:
    redes = st.slider("📱 Minutos en redes", 0, 300, 60)
    dificultad = st.slider("📚 Dificultad (1-5)", 1, 5, 3)

st.markdown("---")

# ----------------------------
# PREDICCIÓN + EXPLICACIÓN
# ----------------------------
if st.button("🔍 Analizar riesgo"):

    redes_log = np.log1p(redes)
    entrada = [[sueno, redes_log, energia, dificultad]]

    if modelo is not None:
        prob = modelo.predict_proba(entrada)[0][1]
        pesos = modelo.coef_[0]
    else:
        prob = 0.5
        pesos = [0,0,0,0]

    # Guardar datos
    nuevo_dato = {
        "sueno": sueno,
        "redes": redes,
        "energia": energia,
        "dificultad": dificultad,
        "procrastino": 1 if prob > 0.5 else 0
    }
    guardar_datos(nuevo_dato)

    # ----------------------------
    # RESULTADO
    # ----------------------------
    st.subheader("📊 Resultado")
    st.progress(int(prob * 100))
    st.metric("Probabilidad de procrastinar", f"{prob*100:.2f}%")

    if prob < 0.4:
        st.success("🟢 Riesgo bajo")
    elif prob < 0.7:
        st.warning("🟡 Riesgo medio")
    else:
        st.error("🔴 Riesgo alto")

    st.markdown("---")

    # ----------------------------
    # 🔥 EXPLICACIÓN DEL MODELO
    # ----------------------------
    st.markdown("### 🧠 ¿Por qué esta predicción?")

    variables = ["Sueño", "Redes", "Energía", "Dificultad"]
    valores = [sueno, redes_log, energia, dificultad]

    contribuciones = []

    for v, p, val in zip(variables, pesos, valores):
        impacto = p * val
        contribuciones.append((v, impacto))

    contribuciones.sort(key=lambda x: abs(x[1]), reverse=True)

    for var, impacto in contribuciones:
        if impacto > 0:
            st.write(f"🔺 {var} aumenta el riesgo")
        elif impacto < 0:
            st.write(f"🔻 {var} reduce el riesgo")

    # ----------------------------
    # FACTORES
    # ----------------------------
    st.markdown("### 🧠 Factores detectados")

    factores = []

    if redes > 120:
        factores.append("uso elevado de redes 📱")
    if sueno < 5:
        factores.append("falta de sueño 😴")
    if energia <= 2:
        factores.append("baja energía ⚡")
    if dificultad >= 4:
        factores.append("tarea difícil 📚")

    if factores:
        st.write(", ".join(factores))
    else:
        st.write("No se detectaron factores críticos")

    # ----------------------------
    # RECOMENDACIONES
    # ----------------------------
    st.markdown("### 💡 Recomendaciones")

    if redes > 120:
        st.write("📱 Reduce el uso de redes antes de estudiar")
    if sueno < 5:
        st.write("😴 Mejora tu descanso")
    if energia <= 2:
        st.write("⚡ Realiza pausas activas")
    if dificultad >= 4:
        st.write("📚 Divide la tarea en partes pequeñas")

    if prob < 0.4:
        st.success("✅ Buen equilibrio de hábitos")

    st.info("🎯 Objetivo: ayudarte a tomar mejores decisiones")

# ----------------------------
# BOTÓN GRÁFICAS
# ----------------------------
if "mostrar" not in st.session_state:
    st.session_state.mostrar = False

if st.button("📊 Ver análisis"):
    st.session_state.mostrar = not st.session_state.mostrar

# ----------------------------
# GRÁFICAS
# ----------------------------
if st.session_state.mostrar:

    st.markdown("## 📊 Análisis de datos")

    fig1, ax1 = plt.subplots()
    conteo = data["procrastino"].value_counts().sort_index()
    ax1.bar(["No", "Sí"], conteo)
    st.pyplot(fig1)

    if modelo is not None:
        fig2, ax2 = plt.subplots()
        ax2.barh(["Sueño", "Redes", "Energía", "Dificultad"], modelo.coef_[0])
        st.pyplot(fig2)
    else:
        st.warning("⚠️ Aún no hay suficientes datos")