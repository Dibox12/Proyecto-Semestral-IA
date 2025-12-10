import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_clean_flowchart():
    # 1. Configuración del Lienzo
    # Aumentamos un poco el tamaño para asegurar que todo quepa sin tight_layout
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 2. Estilos Globales
    c_input = "#dfe6e9"   # Gris claro
    c_proc = "#74b9ff"    # Azul suave
    c_model = "#a29bfe"   # Violeta suave
    c_out = "#55efc4"     # Verde menta
    c_text = "#2d3436"
    
    box_props = dict(boxstyle="round,pad=0.6", ec="#b2bec3", lw=1.5)
    arrow_props = dict(arrowstyle="->", lw=2, color="#636e72")
    
    # --- TÍTULO ---
    ax.text(8, 9.5, "ARQUITECTURA DEL SISTEMA DE PREDICCIÓN", 
            fontsize=18, fontweight='bold', ha='center', color=c_text)

    # --- COLUMNA 1: ENTRADAS (Usuario) ---
    ax.text(2, 8.8, "1. DATOS DE ENTRADA", fontsize=12, fontweight='bold', ha='center', color=c_text)
    
    # Caja Biometría
    ax.text(2, 7.5, "BIOMETRÍA\n(Edad, Peso, Altura,\nFrec. Cardiaca, Sexo)", 
            ha="center", va="center", size=10, bbox=dict(fc=c_input, **box_props))
            
    # Caja Hábitos
    ax.text(2, 3.5, "ESTILO DE VIDA\n(Sueño, Actividad,\nCalorías, Agua, Calidad)", 
            ha="center", va="center", size=10, bbox=dict(fc=c_input, **box_props))

    # --- COLUMNA 2: PROCESAMIENTO (Feature Engineering) ---
    ax.text(6, 8.8, "2. TRANSFORMACIÓN", fontsize=12, fontweight='bold', ha='center', color=c_text)

    # Nodos de proceso
    ax.text(6, 7.5, "Cálculo IMC\n(Índice Masa Corporal)", 
            ha="center", va="center", size=9, bbox=dict(fc=c_proc, **box_props))
            
    ax.text(6, 5.5, "Score Actividad\n(Escala Jurca 1-7)", 
            ha="center", va="center", size=9, bbox=dict(fc=c_proc, **box_props))
            
    ax.text(6, 3.5, "Normalización (Scaler)\n(Escala 0-1 para MLP)", 
            ha="center", va="center", size=9, bbox=dict(fc=c_proc, **box_props))

    # --- COLUMNA 3: MODELO (Cerebro) ---
    ax.text(10, 8.8, "3. INTELIGENCIA ARTIFICIAL", fontsize=12, fontweight='bold', ha='center', color=c_text)

    # Contenedor Grande para el Modelo
    rect = patches.FancyBboxPatch((8.5, 2.0), 3, 6, boxstyle="round,pad=0.2", 
                                  linewidth=2, edgecolor=c_model, facecolor="none", linestyle="--")
    ax.add_patch(rect)
    ax.text(10, 8.2, "Modelo Híbrido", fontsize=10, color=c_model, ha="center")

    # Nodo MLP
    ax.text(10, 6.0, "RED NEURONAL\n(MLP Regressor)\n[128, 64 neuronas]", 
            ha="center", va="center", size=11, fontweight='bold', bbox=dict(fc=c_model, **box_props))
    
    # Nodo Jurca (Feedback visual)
    ax.text(10, 3.0, "Fórmula Jurca\n(Validación Cruzada)", 
            ha="center", va="center", size=9, bbox=dict(fc="#fab1a0", **box_props))

    # --- COLUMNA 4: SALIDA ---
    ax.text(14, 8.8, "4. RESULTADO", fontsize=12, fontweight='bold', ha='center', color=c_text)

    # Nodo Final
    ax.text(14, 5.5, "SCORE RENDIMIENTO\n(0 - 100)", 
            ha="center", va="center", size=14, fontweight='bold', bbox=dict(fc=c_out, ec="green", lw=3, boxstyle="round,pad=1"))


    # --- CONEXIONES (USANDO SOLO ARC3 PARA EVITAR ERRORES) ---
    # Usamos connectionstyle="arc3" que es una curva simple y robusta.

    # 1. Biometría -> IMC (Recto)
    ax.annotate("", xy=(4.9, 7.5), xytext=(3.5, 7.5), arrowprops=arrow_props)
    
    # 2. Biometría -> Jurca (Curva larga hacia abajo)
    ax.annotate("", xy=(8.9, 3.0), xytext=(3.5, 7.3), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#b2bec3", connectionstyle="arc3,rad=-0.3"))

    # 3. Hábitos -> Score Actividad (Curva subiendo)
    ax.annotate("", xy=(4.9, 5.5), xytext=(3.5, 3.8), 
                arrowprops=dict(arrowstyle="->", lw=2, color="#636e72", connectionstyle="arc3,rad=-0.2"))
    
    # 4. Hábitos -> Normalización (Recto)
    ax.annotate("", xy=(4.9, 3.5), xytext=(3.5, 3.5), arrowprops=arrow_props)

    # 5. Procesos -> MLP (Convergen en el centro)
    # IMC -> MLP
    ax.annotate("", xy=(8.8, 6.2), xytext=(7.1, 7.5), 
                arrowprops=dict(arrowstyle="->", lw=2, color="#0984e3", connectionstyle="arc3,rad=-0.2"))
    # Score Act -> MLP
    ax.annotate("", xy=(8.8, 6.0), xytext=(7.1, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0984e3"))
    # Norm -> MLP
    ax.annotate("", xy=(8.8, 5.8), xytext=(7.1, 3.5), 
                arrowprops=dict(arrowstyle="->", lw=2, color="#0984e3", connectionstyle="arc3,rad=0.2"))

    # 6. MLP -> Salida
    ax.annotate("", xy=(12.5, 5.5), xytext=(11.2, 6.0), arrowprops=dict(arrowstyle="->", lw=3, color=c_model))

    # 7. Jurca -> Salida (Feedback punteado)
    ax.annotate("", xy=(12.8, 5.0), xytext=(11.1, 3.0), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#fab1a0", ls="--", connectionstyle="arc3,rad=0.2"))
    ax.text(12, 4.0, "Feedback", fontsize=8, color="#e17055")

    # IMPORTANTE: Eliminamos tight_layout() porque causa conflictos con flechas manuales
    # plt.tight_layout() 
    
    print("Generando imagen 'flujo_limpio.png'...")
    plt.savefig('flujo_limpio.png', dpi=300, bbox_inches='tight') # Usamos bbox_inches aquí en su lugar
    print("Imagen generada exitosamente.")
    plt.show()

if __name__ == "__main__":
    draw_clean_flowchart()