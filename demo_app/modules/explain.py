import shap
import matplotlib.pyplot as plt

def generate_shap_plot(model, df_input, dark_mode=False):
    # Configure theme styling
    if dark_mode:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        text_color = "#ffffff"
    else:
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        text_color = "#000000"

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df_input)
        shap.plots.bar(shap_values[0], max_display=7, show=False)
        # Fix label colors if in dark mode
        plt.gca().xaxis.label.set_color(text_color)
        plt.gca().yaxis.label.set_color(text_color)
        plt.gca().tick_params(colors=text_color)
    except Exception:
        # Fallback to feature importances
        if hasattr(model, "feature_importances_"):
            top_feats = df_input.columns[:7]
            importances = model.feature_importances_[:7]
            ax.barh(top_feats, importances, color="#4da6ff" if dark_mode else "#1f77b4")
            ax.set_title("Relative Feature Importance (MDI)", color=text_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)

    plt.tight_layout()
    return fig