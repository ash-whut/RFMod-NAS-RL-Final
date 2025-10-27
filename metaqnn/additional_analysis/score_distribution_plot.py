import pandas as pd
import matplotlib.pyplot as plt

def score_plot(file_path: str) -> None:
    original_df = pd.read_csv(f"{file_path}/replay_database.csv")

    # Scoring metric
    original_df['model_score'] = (original_df['accuracy'] - 0.25)**3 / original_df['trainable_parameters']**0.25
    original_df = original_df[['model_score', 'ix_q_value_update']]
    
    df_best = original_df.sort_values('model_score', ascending=False).head(10)
    df_everything_else = original_df[~original_df['ix_q_value_update'].isin(df_best['ix_q_value_update'])]

    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'maroon', 'brown', 'teal']
    markers = ['d', 's', '^', 'v', '<', '>', 'D', 'P', '*', 'X']
    
    plt.rcParams['font.size'] = '15'
    plt.figure(figsize = (15,15))
    plt.plot(df_everything_else['ix_q_value_update'], df_everything_else['model_score'], 'bo', markeredgewidth=5.0, label='Other Models')

    for i in range(len(df_best['ix_q_value_update'])):
        plt.plot(df_best['ix_q_value_update'].iloc[i], df_best['model_score'].iloc[i], c=colors[i], marker=markers[i], markersize=12.0, label=f"Model_{i + 1}")

    plt.legend(title="Model ID", loc='lower right')
    plt.tight_layout()
    plt.savefig(f"accuracy_distribution.pdf", format='pdf', dpi=1200, bbox_inches='tight')
    plt.savefig(f"accuracy_distribution.eps", format='eps', dpi=1200, bbox_inches='tight')