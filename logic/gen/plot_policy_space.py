import matplotlib.pyplot as plt
import matplotlib.patches as patches

def fig_policy_space(out_dir):
    """Generate the policy configuration space diagram in a cartographic style."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor='white')
    ax.axis('off')
    
    # Styles
    box_style = dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#0066cc", lw=1.5)
    title_style = dict(fontsize=12, fontweight='bold', color="#1a202c", ha='center', va='center')
    text_style = dict(fontsize=10, color="#4a5568", ha='center', va='center')
    
    # Nodes
    y_top = 0.7
    y_bot = 0.3
    
    # Stage 1
    ax.text(0.15, y_top + 0.15, "STAGE 1\nMandatory Selection", **title_style)
    ax.text(0.15, y_top, "Last-Minute (CF70, CF90)\nLook-Ahead\nService-Level (SL1, SL2)", 
            bbox=box_style, **text_style)
    
    # Stage 2
    ax.text(0.5, y_top + 0.15, "STAGE 2\nRoute Construction", **title_style)
    ax.text(0.5, y_top, "ALNS, BPC, HGS\nACO-HH, PG-CLNS\nPSOMA, SANS, SWC-TCF", 
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#00a859", lw=1.5), **text_style)
    
    # Stage 3
    ax.text(0.85, y_top + 0.15, "STAGE 3\nRoute Improvement", **title_style)
    ax.text(0.85, y_top, "CLS\nFast-TSP", 
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#f26522", lw=1.5), **text_style)
            
    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#bac4ce")
    ax.annotate("", xy=(0.35, y_top), xytext=(0.3, y_top), arrowprops=arrow_props)
    ax.annotate("", xy=(0.70, y_top), xytext=(0.65, y_top), arrowprops=arrow_props)
    
    # Bottom Note
    ax.text(0.5, y_bot, "32 Strategies × 8 Constructors × 33 Improvers\n= 8,448 Configuration Space", 
            fontsize=11, fontweight='bold', color="#1a202c", ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.8", facecolor="#eef2f5", edgecolor="none"))
    
    # Legend/Key
    ax.plot([0.1], [0.1], marker='s', markersize=12, color="#0066cc", linestyle='None')
    ax.text(0.13, 0.1, "Multi-Period Scope", va='center', fontsize=9)
    
    ax.plot([0.45], [0.1], marker='s', markersize=12, color="#00a859", linestyle='None')
    ax.text(0.48, 0.1, "Single-Period Scope", va='center', fontsize=9)
    
    ax.plot([0.8], [0.1], marker='s', markersize=12, color="#f26522", linestyle='None')
    ax.text(0.83, 0.1, "Local Search", va='center', fontsize=9)
    
    out = out_dir / "policy_configuration_space.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")

if __name__ == "__main__":
    from pathlib import Path
    out_dir = Path("assets/papers/Simulation_Framework_for_the_MPVRP_with_Profits_in_Smart_Waste_Collection/Images/Results/Generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_policy_space(out_dir)
