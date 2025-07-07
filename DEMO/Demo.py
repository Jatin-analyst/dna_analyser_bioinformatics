# Enhanced DNA Sequence Analyzer with Professional Visualizations

import streamlit as st
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from typing import List, Dict, Tuple

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# 🧬 Web App Title
st.title("🧬 Professional DNA Sequence Analyzer")
st.markdown("Upload a FASTA file to analyze DNA sequences with comprehensive visualizations: GC content, base composition, mutation analysis, and more.")

# 📤 Upload Box
uploaded_file = st.file_uploader("Choose a FASTA file", type=["fasta", "fa"])

def detect_mutations(seq1: str, seq2: str) -> List[Dict]:
    """Detect point mutations between two sequences."""
    mutations = []
    min_len = min(len(seq1), len(seq2))
    
    for i in range(min_len):
        if seq1[i] != seq2[i]:
            mutations.append({
                "Position": i + 1,
                "Base in Seq1": seq1[i],
                "Base in Seq2": seq2[i],
                "Mutation Type": f"{seq1[i]}→{seq2[i]}"
            })
    return mutations

def calculate_similarity_window(seq1: str, seq2: str, window_size: int = 50) -> Tuple[List[int], List[float]]:
    """Calculate similarity percentage over sliding windows."""
    similarities = []
    positions = []
    min_len = min(len(seq1), len(seq2))
    
    for i in range(0, min_len - window_size + 1, window_size//2):
        window1 = seq1[i:i+window_size]
        window2 = seq2[i:i+window_size]
        
        matches = sum(1 for a, b in zip(window1, window2) if a == b)
        similarity = (matches / len(window1)) * 100
        similarities.append(similarity)
        positions.append(i + window_size//2)
    
    return positions, similarities

def plot_gc_content(df_analysis: pd.DataFrame):
    """Create professional GC content visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_analysis)))
    bars = ax.bar(range(len(df_analysis)), df_analysis["GC %"], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_title("GC Content Distribution Across Sequences", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Sequence ID", fontsize=12, fontweight='bold')
    ax.set_ylabel("GC Content (%)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(df_analysis)))
    ax.set_xticklabels(df_analysis["Sequence ID"], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, gc_val in zip(bars, df_analysis["GC %"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{gc_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% GC (Balanced)')
    ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='40% GC')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='60% GC')
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_base_composition_pie(sequences):
    """Plot base composition pie chart for first sequence."""
    first_seq = str(sequences[0].seq)
    base_counts = [first_seq.count(base) for base in ['A', 'T', 'C', 'G']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(base_counts, labels=['A', 'T', 'C', 'G'],
                                     colors=colors, autopct='%1.1f%%', 
                                     startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
    ax.set_title(f'Base Composition - {sequences[0].id}', fontweight='bold', fontsize=14)
    
    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    return fig

def plot_base_comparison(sequences):
    """Plot base composition comparison across sequences."""
    base_data = []
    for record in sequences:
        seq_str = str(record.seq)
        for base in ['A', 'T', 'C', 'G']:
            base_data.append({
                'Sequence': record.id,
                'Base': base,
                'Percentage': (seq_str.count(base) / len(seq_str)) * 100
            })
    
    df_bases = pd.DataFrame(base_data)
    pivot_data = df_bases.pivot(index='Sequence', columns='Base', values='Percentage')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_data.plot(kind='bar', ax=ax, color=colors, alpha=0.8)
    ax.set_title('Base Composition Comparison', fontweight='bold', fontsize=14)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Sequence ID')
    ax.legend(title='Base', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_sequence_lengths(sequences):
    """Plot sequence length comparison."""
    seq_lengths = [len(str(record.seq)) for record in sequences]
    seq_ids = [record.id for record in sequences]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(seq_ids)), seq_lengths, 
                 color=plt.cm.plasma(np.linspace(0, 1, len(seq_ids))),
                 alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Sequence Length Comparison', fontweight='bold', fontsize=14)
    ax.set_ylabel('Length (bp)')
    ax.set_xlabel('Sequence ID')
    ax.set_xticks(range(len(seq_ids)))
    ax.set_xticklabels(seq_ids, rotation=45, ha='right')
    
    # Add value labels
    for bar, length in zip(bars, seq_lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(seq_lengths)*0.01,
               f'{length:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_similarity_analysis(seq1: str, seq2: str, mutation_data: List[Dict], seq_ids: List[str]):
    """Create similarity curve and mutation density plots."""
    positions, similarities = calculate_similarity_window(seq1, seq2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Similarity curve
    ax1.plot(positions, similarities, linewidth=2.5, color='#2E86AB', alpha=0.8)
    ax1.fill_between(positions, similarities, alpha=0.3, color='#2E86AB')
    
    ax1.set_title(f'Sequence Similarity Analysis\n{seq_ids[0]} vs {seq_ids[1]}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Position in Sequence', fontsize=12)
    ax1.set_ylabel('Similarity (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add average similarity line
    avg_similarity = np.mean(similarities)
    ax1.axhline(y=avg_similarity, color='red', linestyle='--', 
               label=f'Average: {avg_similarity:.1f}%', linewidth=2)
    ax1.legend()
    
    # Mutation density plot
    if mutation_data:
        mutation_positions = [m['Position'] for m in mutation_data]
        hist, bins = np.histogram(mutation_positions, bins=20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax2.bar(bin_centers, hist, width=(bins[1]-bins[0])*0.8, 
                color='#F18F01', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_title('Mutation Density Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Position in Sequence', fontsize=12)
        ax2.set_ylabel('Number of Mutations', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 🔍 If File is Uploaded
if uploaded_file:
    try:
        # Convert uploaded file bytes into text
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        
        # Parse the file using Biopython
        records = list(SeqIO.parse(stringio, "fasta"))

        if not records:
            st.error("No sequences found in the uploaded file. Please check your FASTA format.")
        else:
            # List to store all results
            results = []

            # Analyze each sequence
            for record in records:
                seq = record.seq
                seq_str = str(seq)
                results.append({
                    "Sequence ID": record.id,
                    "Length": len(seq),
                    "GC %": round(gc_fraction(seq) * 100, 2),
                    "A Count": seq_str.count('A'),
                    "T Count": seq_str.count('T'),
                    "C Count": seq_str.count('C'),
                    "G Count": seq_str.count('G'),
                    "Transcription": str(seq.transcribe()),
                    "Reverse Complement": str(seq.reverse_complement())
                })

            # Convert to Pandas DataFrame
            df = pd.DataFrame(results)

            # Show the result in Streamlit
            st.write("### ✅ Analysis Results")
            st.dataframe(df)

            # Download Button
            st.download_button("📥 Download CSV", data=df.to_csv(index=False), 
                             file_name="dna_results.csv", mime="text/csv")

            # Professional Visualizations
            st.write("### 📊 Professional Visualizations")
            
            # 1. GC Content Distribution
            st.write("#### GC Content Distribution")
            fig_gc = plot_gc_content(df)
            st.pyplot(fig_gc)
            
            # 2. Base Composition Analysis
            st.write("#### Base Composition Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Base Composition (First Sequence)**")
                fig_pie = plot_base_composition_pie(records)
                st.pyplot(fig_pie)
            
            with col2:
                if len(records) > 1:
                    st.write("**Base Composition Comparison**")
                    fig_comp = plot_base_comparison(records)
                    st.pyplot(fig_comp)
                else:
                    st.info("Upload multiple sequences to see comparison")
            
            # 3. Sequence Length Comparison
            if len(records) > 1:
                st.write("#### Sequence Length Comparison")
                fig_length = plot_sequence_lengths(records)
                st.pyplot(fig_length)
            
            # 4. Mutation and Similarity Analysis
            if len(records) >= 2:
                st.write("#### Mutation and Similarity Analysis")
                
                seq1_str = str(records[0].seq)
                seq2_str = str(records[1].seq)
                mutation_data = detect_mutations(seq1_str, seq2_str)
                
                # Calculate overall similarity
                min_len = min(len(seq1_str), len(seq2_str))
                matches = sum(1 for a, b in zip(seq1_str, seq2_str) if a == b)
                overall_similarity = (matches / min_len) * 100
                
                # Display mutation summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Mutations", len(mutation_data))
                with col2:
                    st.metric("Mutation Rate", f"{len(mutation_data)/min_len*100:.2f}%")
                with col3:
                    st.metric("Overall Similarity", f"{overall_similarity:.2f}%")
                
                # Show mutation table if mutations found
                if mutation_data:
                    st.write("**Mutation Details**")
                    df_mutations = pd.DataFrame(mutation_data)
                    st.dataframe(df_mutations)
                    
                    # Download mutation data
                    st.download_button("📥 Download Mutation Report", 
                                     data=df_mutations.to_csv(index=False), 
                                     file_name="mutation_report.csv", 
                                     mime="text/csv")
                
                # Similarity analysis plot
                if min_len > 100:
                    st.write("**Similarity Curve and Mutation Density**")
                    seq_ids = [record.id for record in records[:2]]
                    fig_sim = plot_similarity_analysis(seq1_str, seq2_str, mutation_data, seq_ids)
                    st.pyplot(fig_sim)
                else:
                    st.info("Sequences too short for sliding window analysis")
            
            # Summary statistics
            st.write("### 📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Sequences", len(records))
            with col2:
                avg_length = df['Length'].mean()
                st.metric("Average Length", f"{avg_length:.0f} bp")
            with col3:
                avg_gc = df['GC %'].mean()
                st.metric("Average GC%", f"{avg_gc:.1f}%")
            with col4:
                total_bases = df['Length'].sum()
                st.metric("Total Bases", f"{total_bases:,}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file is a valid FASTA format.")