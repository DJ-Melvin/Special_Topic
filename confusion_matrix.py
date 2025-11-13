#!/usr/bin/env python3
"""
Circuit-Level Confusion Matrix Generator
Based on Competition Results
"""

import numpy as np
import matplotlib.pyplot as plt

# Competition results scores
# Cases 0-39: Trojan cases (score >= 2 means correctly detected)
# Cases 40-59: Trojan-free cases (score = 2 means correctly identified as clean, score = 0 means false positive)
score = [2.2526, 2.3843, 0, 0, 2.1984, 2.3226, 2.1976, 2.594, 2.6652, 2.6652, 2.1459, 2.1449, 2.1481, 2.2615, 2.2599, 2.955, 2.9548, 2.8925, 2.8928, 2.4511, 2, 3, 2.7723, 2.8158, 2.9535, 0, 2.896, 2.9383, 2.9154, 0, 2.6154, 2.814, 2.7652, 2.814, 2.7652, 2.9618, 2.8899, 2.9812, 0, 2.975, 2.8, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2]

def analyze_circuit_results():
    """
    Analyze circuit-level results and create confusion matrix.
    """
    # Initialize counters
    TP = TN = FP = FN = 0
    
    # Detailed analysis for each case
    trojan_cases = []
    free_cases = []
    
    # Analyze cases 0-39 (Trojan cases)
    for case_id in range(40):
        case_score = score[case_id]
        if case_score >= 2:
            TP += 1  # True Positive: Correctly detected trojan
            status = "TP"
        else:
            FN += 1  # False Negative: Missed trojan
            status = "FN"
        
        trojan_cases.append({
            'case_id': case_id,
            'score': case_score,
            'status': status
        })
    
    # Analyze cases 40-59 (Trojan-free cases)
    for case_id in range(40, 60):
        case_score = score[case_id]
        if case_score == 2:
            TN += 1  # True Negative: Correctly identified as clean
            status = "TN"
        else:
            FP += 1  # False Positive: Incorrectly flagged as trojan
            status = "FP"
        
        free_cases.append({
            'case_id': case_id,
            'score': case_score,
            'status': status
        })
    
    return TP, TN, FP, FN, trojan_cases, free_cases

def create_confusion_matrix_plot(TP, TN, FP, FN, trojan_cases, free_cases):
    """
    Create comprehensive confusion matrix visualization.
    """
    # Calculate metrics
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create confusion matrix with TP in top-left
    cm = np.array([[TP, FN], [FP, TN]])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Circuit-Level Hardware Trojan Detection Results\n(Competition Performance Analysis)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Main Confusion Matrix
    ax1 = axes[0, 0]
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax1.figure.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    
    # Add text annotations with TP in top-left
    thresh = cm.max() / 2.
    labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{labels[i][j]}\n{cm[i, j]}',
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=16, fontweight='bold')
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5) 
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Trojan', 'Trojan-free'], fontsize=12)
    ax1.set_yticklabels(['Trojan', 'Trojan-free'], fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Invert Y axis to put TP in top-left
    
    # 2. Metrics Panel
    ax2 = axes[0, 1]
    ax2.axis('off')
    metrics_text = f"""PERFORMANCE METRICS

Accuracy: {accuracy:.3f} ({TP+TN}/{total})
Precision: {precision:.3f} ({TP}/{TP+FP})
Recall: {recall:.3f} ({TP}/{TP+FN})
Specificity: {specificity:.3f} ({TN}/{TN+FP})
F1-Score: {f1_score:.3f}

CONFUSION MATRIX:
• True Positives (TP): {TP}
  Trojans correctly detected
• True Negatives (TN): {TN}  
  Clean circuits correctly identified
• False Positives (FP): {FP}
  Clean circuits wrongly flagged
• False Negatives (FN): {FN}
  Trojans missed

DETECTION RATES:
• Trojan Detection: {TP}/40 = {TP/40:.1%}
• Clean Identification: {TN}/20 = {TN/20:.1%}
"""
    
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Trojan Cases Score Distribution
    ax3 = axes[1, 0]
    case_ids = [case['case_id'] for case in trojan_cases]
    scores = [case['score'] for case in trojan_cases]
    colors = ['darkgreen' if case['status'] == 'TP' else 'darkred' for case in trojan_cases]
    
    bars = ax3.bar(case_ids, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=2, color='black', linestyle='--', linewidth=2, label='Threshold (Score = 2)')
    ax3.set_xlabel('Case ID (Trojan Cases 0-39)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Trojan Cases: Score Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    ax3.text(0.02, 0.98, f'Detected: {TP}/40\nMissed: {FN}/40', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. Trojan-Free Cases Results
    ax4 = axes[1, 1]
    free_case_ids = [case['case_id'] for case in free_cases]
    free_scores = [case['score'] for case in free_cases]
    free_colors = ['darkgreen' if case['status'] == 'TN' else 'darkred' for case in free_cases]
    
    x_pos = range(len(free_case_ids))
    bars = ax4.bar(x_pos, free_scores, color=free_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Case Index (Cases 40-59)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Trojan-Free Cases Results', fontsize=12, fontweight='bold')
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_yticks([0, 2])
    ax4.set_yticklabels(['Wrong (0)', 'Correct (2)'])
    ax4.grid(True, alpha=0.3)
    
    # Set x-axis labels
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([str(cid) for cid in free_case_ids], rotation=45)
    
    # Add statistics
    ax4.text(0.02, 0.98, f'Correct: {TN}/20\nWrong: {FP}/20', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: figures/confusion_matrix.png")
    plt.show()
    
    return accuracy, precision, recall, specificity, f1_score

def print_detailed_results(TP, TN, FP, FN, trojan_cases, free_cases):
    """
    Print detailed analysis of results.
    """
    print("\n" + "="*80)
    print("DETAILED COMPETITION RESULTS ANALYSIS")
    print("="*80)
    
    # Overall statistics
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total Cases: {total}")
    print(f"  Correct Classifications: {TP + TN}")
    print(f"  Incorrect Classifications: {FP + FN}")
    print(f"  Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Trojan cases analysis
    print(f"\nTROJAN CASES (0-39) ANALYSIS:")
    print(f"  Successfully Detected (TP): {TP}/40")
    print(f"  Missed (FN): {FN}/40")
    print(f"  Detection Rate: {TP/40:.3f} ({TP/40*100:.1f}%)")
    
    # Show missed trojans
    missed_trojans = [case for case in trojan_cases if case['status'] == 'FN']
    if missed_trojans:
        missed_ids = [case['case_id'] for case in missed_trojans]
        print(f"  Missed Trojan Cases: {missed_ids}")
        avg_missed_score = np.mean([case['score'] for case in missed_trojans])
        print(f"  Average Score of Missed Cases: {avg_missed_score:.3f}")
    
    # Trojan-free cases analysis  
    print(f"\nTROJAN-FREE CASES (40-59) ANALYSIS:")
    print(f"  Correctly Identified (TN): {TN}/20")
    print(f"  False Alarms (FP): {FP}/20")
    print(f"  Specificity: {TN/20:.3f} ({TN/20*100:.1f}%)")
    
    # Show false alarms
    false_alarms = [case for case in free_cases if case['status'] == 'FP']
    if false_alarms:
        false_alarm_ids = [case['case_id'] for case in false_alarms]
        print(f"  False Alarm Cases: {false_alarm_ids}")
    
    # Score distribution analysis
    print(f"\nSCORE DISTRIBUTION:")
    trojan_scores = [case['score'] for case in trojan_cases]
    print(f"  Trojan Cases - Min: {min(trojan_scores):.3f}, Max: {max(trojan_scores):.3f}, Avg: {np.mean(trojan_scores):.3f}")
    
    successful_scores = [case['score'] for case in trojan_cases if case['status'] == 'TP']
    if successful_scores:
        print(f"  Successful Detections - Min: {min(successful_scores):.3f}, Max: {max(successful_scores):.3f}, Avg: {np.mean(successful_scores):.3f}")

def main():
    """
    Main function to generate confusion matrix from competition results.
    """
    print("="*80)
    print("CIRCUIT-LEVEL CONFUSION MATRIX GENERATOR")
    print("Based on Competition Results")
    print("="*80)
    
    # Analyze results
    TP, TN, FP, FN, trojan_cases, free_cases = analyze_circuit_results()
    
    # Generate visualization
    accuracy, precision, recall, specificity, f1_score = create_confusion_matrix_plot(
        TP, TN, FP, FN, trojan_cases, free_cases
    )
    
    # Print detailed analysis
    print_detailed_results(TP, TN, FP, FN, trojan_cases, free_cases)
    
    # Final summary
    print(f"\n" + "="*80)
    print("FINAL METRICS SUMMARY")
    print("="*80)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"             Predicted")
    print(f"          Trojan  Clean")
    print(f"Actual Trojan  {TP:2d}     {FN:2d}")
    print(f"       Clean   {FP:2d}     {TN:2d}")

if __name__ == "__main__":
    main()
