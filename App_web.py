import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Predictor de Procrastinación", page_icon="🧠")

st.title("🧠 Sistema Inteligente de Predicción de Procrastinación")

# ----------------------------
# FUNCIÓN PARA GUARDAR DATOS
# ----------------------------
def guardar_datos(nuevo_dato):
    archivo = "datos_usuarios.csv"

    if os.path.exists(archivo) and os.path.getsize(archivo) > 0:
        try:
            df = pd.read_csv(archivo)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([nuevo_dato])], ignore_index=True)
    df.to_csv(archivo, index=False)

# ----------------------------
# DATOS INICIALES
# ----------------------------
if os.path.exists("datos_usuarios.csv") and os.path.getsize("datos_usuarios.csv") > 0:
    data = pd.read_csv("datos_usuarios.csv")
else:
    data = pd.DataFrame({
        "sueno": [4,6,7,5,8,3,6,7,5,4],
        "redes": [120,60,30,90,20,150,80,40,100,110],
        "energia": [2,3,4,2,5,1,3,4,2,2],
        "dificultad": [5,3,2,4,1,5,3,2,4,5],
        "procrastino": [1,0,0,1,0,1,1,0,1,1]
    })

X = data[["sueno", "redes", "energia", "dificultad"]]
y = data["procrastino"]

if len(y.unique()) > 1:
    modelo = LogisticRegression()
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
    redes = st.slider("📱 Minutos en redes", 0, 180, 60)
    dificultad = st.slider("📚 Dificultad (1-5)", 1, 5, 3)

st.markdown("---")

# ----------------------------
# PREDICCIÓN + GUARDADO
# ----------------------------
if st.button("🔍 Analizar riesgo"):

    entrada = [[sueno, redes, energia, dificultad]]

    # Predicción segura
    if modelo is not None:
        prob = modelo.predict_proba(entrada)[0][1]
    else:
        prob = 0.5  # valor neutro

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
    # FACTORES
    # ----------------------------
    st.markdown("### 🧠 Factores detectados")

    factores = []

    if redes > 90:
        factores.append("uso elevado de redes sociales 📱")
    if sueno < 5:
        factores.append("falta de sueño 😴")
    if energia <= 2:
        factores.append("bajo nivel de energía ⚡")
    if dificultad >= 4:
        factores.append("alta dificultad de la tarea 📚")

    if factores:
        st.write("Factores que influyen:", ", ".join(factores))
    else:
        st.write("No se detectaron factores críticos.")

    # ----------------------------
    # EVALUACIÓN AVANZADA
    # ----------------------------
    st.markdown("### ⚠️ Evaluación avanzada")

    if redes > 90 and sueno < 5:
        st.error("🚨 Exceso de redes + poco sueño → Riesgo muy alto")

    if energia <= 2 and dificultad >= 4:
        st.warning("⚠️ Baja energía + tarea difícil → Posible evasión")

    if prob > 0.75:
        st.error("🔥 Alta probabilidad de procrastinación detectada")

    # ----------------------------
    # RECOMENDACIONES
    # ----------------------------
    st.markdown("### 💡 Recomendaciones")

    if redes > 90:
        st.write("📱 Reduce el uso de redes antes de estudiar")

    if sueno < 5:
        st.write("😴 Mejora tus horas de descanso")

    if energia <= 2:
        st.write("⚡ Realiza pausas activas")

    if dificultad >= 4:
        st.write("📚 Divide la tarea en partes pequeñas")

    if prob < 0.4:
        st.success("✅ Buen equilibrio de hábitos")

    st.markdown("---")
    st.info("🎯 Objetivo: ayudarte a tomar mejores decisiones antes de procrastinar")
# ----------------------------
# BOTÓN VER ANÁLISIS
# ----------------------------
if "mostrar_graficas" not in st.session_state:
    st.session_state.mostrar_graficas = False

if st.button("📊 Ver análisis de datos"):
    st.session_state.mostrar_graficas = not st.session_state.mostrar_graficas

# ----------------------------
# GRÁFICAS
# ----------------------------
if st.session_state.mostrar_graficas:

    st.markdown("## 📊 Análisis de datos")

    # --------- Distribución ----------
    fig1, ax1 = plt.subplots()
    conteo = data["procrastino"].value_counts().sort_index()
    labels = ["No procrastina", "Sí procrastina"]

    ax1.bar(labels, conteo, color=["green", "red"])

    for i, v in enumerate(conteo):
        ax1.text(i, v + 0.1, str(v), ha='center')

    st.pyplot(fig1)

    # --------- Scatter ----------
    fig2, ax2 = plt.subplots()

    no = data[data["procrastino"] == 0]
    si = data[data["procrastino"] == 1]

    ax2.scatter(no["redes"], no["procrastino"], color="green", label="No")
    ax2.scatter(si["redes"], si["procrastino"], color="red", label="Sí")

    ax2.legend()
    st.pyplot(fig2)

    # --------- Importancia ----------
    fig3, ax3 = plt.subplots()

    variables = X.columns
    pesos = modelo.coef_[0]

    ax3.barh(variables, pesos)

    st.pyplot(fig3)