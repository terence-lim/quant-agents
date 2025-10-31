"""
import matplotlib.pyplot as plt
import seaborn as sns

data = [44, 45, 40, 41, 39]
labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

# declaring exploding pie
explode = [0, 0.1, 0, 0, 0]

# define Seaborn color palette to use
colors = sns.color_palette('dark')

# plotting data on chart
plt.pie(data, labels=labels, colors=colors, explode=explode, autopct='%.0f%%')
plt.show()
"""

country_dict = {
    'USA': "United States",
    "JPN": "Japan",
    "GBR": "United Kingdom",
    "CAN": "Canada",
    "FRA": "France",
    "DEU": "Germany",
    "AUS": "Australia",
    "CHE": "Switzerland",
    "NLD": "Netherlands",
    "HKG": "Hong Kong",
    "ESP": "Spain",
    "SWE": "Sweden",
    "DNK": "Denmark",
    "ITA": "Italy",
    "BEL": "Belgium",
    "SGP": "Singapore",
    "NOR": "Norway",
    "FIN": "Finland",
    "ISR": "Israel",
    "AUT": "Austria",
    "IRL": "Ireland",
    "NZL": "New Zealand",
    "PRT": "Portugal",
}

"""
TODO:
- def TimeFrame.performance(): annualized return, volatility, sharpe ratio, max drawdown
- def PanelFrame.turnover()
- Download and check JKP factors
"""

import matplotlib.pyplot as plt
import seaborn as sns

def pie_chart(data, labels, title: str, ncol=3):
    """Plot a pie chart with the given data and labels."""
    colors = sns.color_palette('pastel')

    fig, ax = plt.subplots(figsize=(8, 6))

    wedge_width = 0.5
    pctdistance = 1 - wedge_width / 2

    wedges, texts, autotexts = ax.pie(
        data,
        labels=labels,                 # show labels on wedges
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=wedge_width),
        autopct=lambda pct: f"{pct:.0f}%",
        pctdistance=pctdistance
    )

    # --- Adjust label styles ---
    for text in texts:
        text.set_fontsize(16)          # category labels (outside wedges)
#        text.set_weight('bold')
        text.set_color('black')

    for autotext in autotexts:
        autotext.set_fontsize(14)        # increase font size
        autotext.set_color('black')
#        autotext.set_weight('bold')

    # bold title, large font
    ax.set_title(title, fontweight='bold', fontsize=18)

    # Adjust legend placement: lower center, wide layout
    ax.legend(
        wedges,
        labels,
        fontsize=14,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),  # center below chart
        ncol=ncol,
        #ncol=(len(labels)+1)//2,            # all items in one row
        frameon=False
    )

if __name__ == "__main__":
    data = {"GPT family":58, "Claude":13, "LLaMA":11, "Gemini": 11, "other":7}
    pie_chart(data.values(), data.keys(), title="Models Employed (%)")
    plt.tight_layout()
    plt.savefig("models.svg", bbox_inches='tight')

    data = {"Social media": 25, "Education": 21, "Software": 20, "Healthcare": 16, "Arts/Humanities": 18}
    pie_chart(data.values(), data.keys(), title="Datasets by Domain (%)")
    plt.tight_layout()
    plt.savefig("domain.svg", bbox_inches='tight')

    data = {"Zero-shot":35, "Few-shot":16, "Chain-of-thought": 13, "Reflexion":15, "Tool/agent":11}
    pie_chart(data.values(), data.keys(), title="Prompting Strategies (%)")
    plt.tight_layout()
    plt.savefig("prompting.svg", bbox_inches='tight')

    data = {"Human review":40, "Similarity metrics":27, "Task-based":13, "hybrid":20}
    pie_chart(data.values(), data.keys(), title="Evaluation Fragmentation (%)")
    plt.tight_layout()
    plt.savefig("evaluation.svg", bbox_inches='tight')

    data = dict(Inductive=64, Hybrid=22, Deductive=9)
    pie_chart(data.values(), data.keys(), title="Type of Thematic Analysis (%)")
    plt.tight_layout()
    plt.savefig("TA.svg", bbox_inches='tight')
    
    plt.show()


