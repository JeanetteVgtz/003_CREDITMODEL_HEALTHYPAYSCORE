"""
Professional Architecture Diagram
Clean, modern style matching previous diagrams
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.patches as mpatches

def create_professional_architecture():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(8, 9.2, 'CREDIT RISK ASSESSMENT SYSTEM',
            ha='center', fontsize=22, fontweight='bold', family='sans-serif')
    ax.text(8, 8.7, 'Neural Network-Based Automated Credit Decision Platform',
            ha='center', fontsize=12, style='italic', color='#555')

    # Main flow boxes - larger, more prominent
    box_height = 1.8
    box_width = 3.2
    y_position = 5.5

    # Color scheme (professional blues and greens)
    colors = {
        'input': '#3498DB',
        'preprocess': '#E67E22',
        'model': '#9B59B6',
        'decision': '#F39C12',
        'output': '#27AE60'
    }

    # Box 1: INPUT
    input_box = FancyBboxPatch((0.8, y_position), box_width, box_height,
                              boxstyle="round,pad=0.15",
                              edgecolor=colors['input'], facecolor=colors['input'],
                              linewidth=3, alpha=0.85)
    ax.add_patch(input_box)
    ax.text(2.4, y_position + 1.4, 'DATA INPUT', ha='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(2.4, y_position + 1.0, 'CSV Upload', ha='center', fontsize=11, color='white')
    ax.text(2.4, y_position + 0.7, '87 Features', ha='center', fontsize=11, color='white')
    ax.text(2.4, y_position + 0.4, '13,184 Applications', ha='center', fontsize=9, color='white', style='italic')

    # Box 2: PREPROCESSING
    prep_box = FancyBboxPatch((4.4, y_position), box_width, box_height,
                             boxstyle="round,pad=0.15",
                             edgecolor=colors['preprocess'], facecolor=colors['preprocess'],
                             linewidth=3, alpha=0.85)
    ax.add_patch(prep_box)
    ax.text(6.0, y_position + 1.4, 'PREPROCESSING', ha='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(6.0, y_position + 1.0, 'Feature Engineering', ha='center', fontsize=11, color='white')
    ax.text(6.0, y_position + 0.7, 'Standardization', ha='center', fontsize=11, color='white')
    ax.text(6.0, y_position + 0.4, 'Missing Value Handling', ha='center', fontsize=9, color='white', style='italic')

    # Box 3: ML MODEL
    model_box = FancyBboxPatch((8.0, y_position), box_width, box_height,
                              boxstyle="round,pad=0.15",
                              edgecolor=colors['model'], facecolor=colors['model'],
                              linewidth=3, alpha=0.85)
    ax.add_patch(model_box)
    ax.text(9.6, y_position + 1.4, 'ML MODEL', ha='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(9.6, y_position + 1.0, 'Neural Network', ha='center', fontsize=11, color='white')
    ax.text(9.6, y_position + 0.7, 'AUC-ROC: 0.794', ha='center', fontsize=11, color='white')
    ax.text(9.6, y_position + 0.4, 'Accuracy: 83.7%', ha='center', fontsize=9, color='white', style='italic')

    # Box 4: DECISION ENGINE
    decision_box = FancyBboxPatch((11.6, y_position), box_width, box_height,
                                 boxstyle="round,pad=0.15",
                                 edgecolor=colors['decision'], facecolor=colors['decision'],
                                 linewidth=3, alpha=0.85)
    ax.add_patch(decision_box)
    ax.text(13.2, y_position + 1.4, 'DECISION', ha='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(13.2, y_position + 1.0, 'Credit Score', ha='center', fontsize=11, color='white')
    ax.text(13.2, y_position + 0.7, 'Interest Rate', ha='center', fontsize=11, color='white')
    ax.text(13.2, y_position + 0.4, 'Loan Amount', ha='center', fontsize=9, color='white', style='italic')

    # Large arrows between boxes
    arrow_props = dict(arrowstyle='->', lw=4, color='#2C3E50')

    arrow1 = FancyArrowPatch((4.0, y_position + 0.9), (4.4, y_position + 0.9),
                            **arrow_props)
    ax.add_artist(arrow1)

    arrow2 = FancyArrowPatch((7.6, y_position + 0.9), (8.0, y_position + 0.9),
                            **arrow_props)
    ax.add_artist(arrow2)

    arrow3 = FancyArrowPatch((11.2, y_position + 0.9), (11.6, y_position + 0.9),
                            **arrow_props)
    ax.add_artist(arrow3)

    # Supporting components section
    ax.text(8, 4.2, 'SUPPORTING COMPONENTS', ha='center',
            fontsize=13, fontweight='bold', color='#2C3E50')

    # Grid of supporting components - with proper spacing
    support_y = 2.4
    support_height = 1.2
    support_width = 3.2
    gap = 0.3  # Gap between boxes

    # Validation
    val_box = FancyBboxPatch((0.8, support_y), support_width, support_height,
                            boxstyle="round,pad=0.1",
                            edgecolor='#34495E', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(val_box)
    ax.text(2.4, support_y + 0.85, 'VALIDATION', ha='center',
            fontsize=11, fontweight='bold', color='#2C3E50')
    ax.text(2.4, support_y + 0.5, '5-Fold Cross-Validation', ha='center', fontsize=9)
    ax.text(2.4, support_y + 0.2, 'AUC: 0.7862 ± 0.0062', ha='center', fontsize=8, style='italic')

    # Fairness
    fair_x = 0.8 + support_width + gap
    fair_box = FancyBboxPatch((fair_x, support_y), support_width, support_height,
                             boxstyle="round,pad=0.1",
                             edgecolor='#34495E', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(fair_box)
    ax.text(fair_x + support_width/2, support_y + 0.85, 'FAIRNESS', ha='center',
            fontsize=11, fontweight='bold', color='#2C3E50')
    ax.text(fair_x + support_width/2, support_y + 0.5, 'Bias Testing', ha='center', fontsize=9)
    ax.text(fair_x + support_width/2, support_y + 0.2, 'Demographic Parity: PASS', ha='center', fontsize=8, style='italic')

    # Threshold
    thresh_x = fair_x + support_width + gap
    thresh_box = FancyBboxPatch((thresh_x, support_y), support_width, support_height,
                               boxstyle="round,pad=0.1",
                               edgecolor='#34495E', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(thresh_box)
    ax.text(thresh_x + support_width/2, support_y + 0.85, 'THRESHOLD', ha='center',
            fontsize=11, fontweight='bold', color='#2C3E50')
    ax.text(thresh_x + support_width/2, support_y + 0.5, 'Adjustable Risk Level', ha='center', fontsize=9)
    ax.text(thresh_x + support_width/2, support_y + 0.2, 'Precision vs Recall Trade-off', ha='center', fontsize=8, style='italic')

    # Deployment
    deploy_x = thresh_x + support_width + gap
    deploy_box = FancyBboxPatch((deploy_x, support_y), support_width, support_height,
                               boxstyle="round,pad=0.1",
                               edgecolor='#34495E', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(deploy_box)
    ax.text(deploy_x + support_width/2, support_y + 0.85, 'DEPLOYMENT', ha='center',
            fontsize=11, fontweight='bold', color='#2C3E50')
    ax.text(deploy_x + support_width/2, support_y + 0.5, 'Streamlit Web App', ha='center', fontsize=9)
    ax.text(deploy_x + support_width/2, support_y + 0.2, 'Real-time Processing', ha='center', fontsize=8, style='italic')

    # Performance metrics box at bottom
    perf_box = FancyBboxPatch((1.5, 0.4), 13, 1.2,
                             boxstyle="round,pad=0.15",
                             edgecolor='#27AE60', facecolor='#D5F4E6', linewidth=3)
    ax.add_patch(perf_box)

    ax.text(8, 1.3, 'SYSTEM PERFORMANCE', ha='center',
            fontsize=12, fontweight='bold', color='#27AE60')

    # Three columns of metrics
    ax.text(3.5, 0.85, 'Processing Speed', ha='center', fontsize=10, fontweight='bold')
    ax.text(3.5, 0.6, '0.002 sec/application', ha='center', fontsize=11, color='#27AE60', fontweight='bold')

    ax.text(8, 0.85, 'Model Accuracy', ha='center', fontsize=10, fontweight='bold')
    ax.text(8, 0.6, 'AUC-ROC: 0.7939 | Acc: 83.7%', ha='center', fontsize=11, color='#27AE60', fontweight='bold')

    ax.text(12.5, 0.85, 'Throughput', ha='center', fontsize=10, fontweight='bold')
    ax.text(12.5, 0.6, '600x faster than manual', ha='center', fontsize=11, color='#27AE60', fontweight='bold')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating professional architecture diagram...")
    fig = create_professional_architecture()

    output_path = "diagrams/architecture_professional.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    plt.close()
    print("\n✅ Professional diagram complete!")