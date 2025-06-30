#!pip install biopython seaborn
import Bio
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional

class DNAAnalyzer:
    """Professional DNA sequence analysis toolkit."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the analyzer with output directory."""
        self.output_dir = output_dir
        self.sequences = []
        self.analysis_results = []
        self._setup_plotting_style()
        self._ensure_output_dir()
    
    def _setup_plotting_style(self):
        """Configure professional plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_sequences(self, fasta_file: Optional[str] = None) -> bool:
        """Load DNA sequences from FASTA file."""
        if not fasta_file:
            fasta_file = input("Enter FASTA file path (or press Enter for 'Sequence.fasta'): ")
            if not fasta_file:
                fasta_file = 'Sequence.fasta'
        
        try:
            self.sequences = list(SeqIO.parse(fasta_file, "fasta"))
            print(f"✓ Loaded {len(self.sequences)} sequences from {fasta_file}")
            return True
        except FileNotFoundError:
            print(f"❌ File '{fasta_file}' not found!")
            return False
    
    def analyze_sequences(self) -> pd.DataFrame:
        """Perform comprehensive analysis on all sequences."""
        self.analysis_results = []
        
        for record in self.sequences:
            seq_str = str(record.seq)
            analysis = {
                "Sequence ID": record.id,
                "Length": len(seq_str),
                "GC %": round(gc_fraction(record.seq) * 100, 2),
                "A Count": seq_str.count('A'),
                "T Count": seq_str.count('T'),
                "C Count": seq_str.count('C'),
                "G Count": seq_str.count('G'),
                "Transcription": str(record.seq.transcribe()),
                "Reverse Complement": str(record.seq.reverse_complement())
            }
            self.analysis_results.append(analysis)
        
        df_analysis = pd.DataFrame(self.analysis_results)
        df_analysis.to_csv(f"{self.output_dir}/dna_analysis.csv", index=False)
        print("✓ DNA Analysis saved to outputs/dna_analysis.csv")
        return df_analysis
    
    def plot_gc_content(self, df_analysis: pd.DataFrame):
        """Create professional GC content visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_analysis)))
        bars = ax.bar(range(len(df_analysis)), df_analysis["GC %"], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_title("GC Content Distribution Across Sequences", 
                    fontsize=16, fontweight='bold', pad=20)
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
        self._add_gc_reference_lines(ax)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gc_content_professional.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _add_gc_reference_lines(self, ax):
        """Add GC content reference lines to plot."""
        reference_lines = [
            (50, 'red', '50% GC (Balanced)'),
            (40, 'orange', '40% GC'),
            (60, 'orange', '60% GC')
        ]
        
        for y_val, color, label in reference_lines:
            linestyle = '--' if y_val == 50 else '--'
            alpha = 0.7 if y_val == 50 else 0.5
            ax.axhline(y=y_val, color=color, linestyle=linestyle, 
                      alpha=alpha, label=label)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def detect_mutations(self, seq1: str, seq2: str) -> List[Dict]:
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
    
    def calculate_similarity_window(self, seq1: str, seq2: str, 
                                  window_size: int = 50) -> Tuple[List[int], List[float]]:
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
    
    def analyze_mutations_and_similarity(self) -> Optional[pd.DataFrame]:
        """Perform mutation and similarity analysis between first two sequences."""
        if len(self.sequences) < 2:
            print("⚠️  Need at least 2 sequences to compare mutations and similarity.")
            return None
        
        seq1_str = str(self.sequences[0].seq)
        seq2_str = str(self.sequences[1].seq)
        mutation_data = self.detect_mutations(seq1_str, seq2_str)
        
        # Calculate overall similarity
        min_len = min(len(seq1_str), len(seq2_str))
        matches = sum(1 for a, b in zip(seq1_str, seq2_str) if a == b)
        overall_similarity = (matches / min_len) * 100
        
        # Save mutation data if found
        df_mutations = None
        if mutation_data:
            df_mutations = pd.DataFrame(mutation_data)
            df_mutations.to_csv(f"{self.output_dir}/mutation_report.csv", index=False)
            print(f"✓ Found {len(mutation_data)} mutations - saved to mutation_report.csv")
            
            self._print_mutation_summary(mutation_data, min_len, overall_similarity)
        else:
            print("✓ No mutations detected between first two sequences.")
        
        # Create similarity curve plot
        if min_len > 100:
            self.plot_similarity_analysis(seq1_str, seq2_str, mutation_data)
        
        return df_mutations
    
    def _print_mutation_summary(self, mutation_data: List[Dict], 
                               min_len: int, overall_similarity: float):
        """Print formatted mutation analysis summary."""
        print(f"\n📊 Mutation Analysis Summary:")
        print(f"   • Total mutations: {len(mutation_data)}")
        print(f"   • Mutation rate: {len(mutation_data)/min_len*100:.2f}%")
        print(f"   • Overall similarity: {overall_similarity:.2f}%")
    
    def plot_similarity_analysis(self, seq1: str, seq2: str, 
                               mutation_data: Optional[List[Dict]] = None):
        """Create similarity curve and mutation density plots."""
        positions, similarities = self.calculate_similarity_window(seq1, seq2)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Similarity curve
        self._plot_similarity_curve(ax1, positions, similarities)
        
        # Mutation density plot
        if mutation_data:
            self._plot_mutation_density(ax2, mutation_data)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/similarity_analysis_professional.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_similarity_curve(self, ax, positions: List[int], similarities: List[float]):
        """Plot the similarity curve."""
        ax.plot(positions, similarities, linewidth=2.5, color='#2E86AB', alpha=0.8)
        ax.fill_between(positions, similarities, alpha=0.3, color='#2E86AB')
        
        ax.set_title(f'Sequence Similarity Analysis\n{self.sequences[0].id} vs {self.sequences[1].id}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Position in Sequence', fontsize=12)
        ax.set_ylabel('Similarity (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add average similarity line
        avg_similarity = np.mean(similarities)
        ax.axhline(y=avg_similarity, color='red', linestyle='--', 
                  label=f'Average: {avg_similarity:.1f}%', linewidth=2)
        ax.legend()
    
    def _plot_mutation_density(self, ax, mutation_data: List[Dict]):
        """Plot mutation density distribution."""
        mutation_positions = [m['Position'] for m in mutation_data]
        hist, bins = np.histogram(mutation_positions, bins=20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax.bar(bin_centers, hist, width=(bins[1]-bins[0])*0.8, 
               color='#F18F01', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title('Mutation Density Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position in Sequence', fontsize=12)
        ax.set_ylabel('Number of Mutations', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def plot_comprehensive_base_analysis(self):
        """Create comprehensive base composition analysis."""
        if not self.sequences:
            print("❌ No sequences loaded for base analysis.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Base Composition Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Pie chart for first sequence
        self._plot_base_pie_chart(axes[0,0])
        
        # 2. Base composition comparison (if multiple sequences)
        if len(self.sequences) > 1:
            self._plot_base_comparison(axes[0,1])
            self._plot_sequence_lengths(axes[1,0])
            self._plot_gc_at_distribution(axes[1,1])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comprehensive_base_analysis_professional.png", 
                   dpi=300, bbox_inches='tight')
        
        try:
            plt.show()
        except:
            print("✓ Charts saved - cannot display in this environment")
    
    def _plot_base_pie_chart(self, ax):
        """Plot base composition pie chart for first sequence."""
        first_seq = str(self.sequences[0].seq)
        base_counts = [first_seq.count(base) for base in ['A', 'T', 'C', 'G']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        wedges, texts, autotexts = ax.pie(base_counts, labels=['A', 'T', 'C', 'G'],
                                         colors=colors, autopct='%1.1f%%', 
                                         startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
        ax.set_title(f'Base Composition\n{self.sequences[0].id}', fontweight='bold')
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_base_comparison(self, ax):
        """Plot base composition comparison across sequences."""
        base_data = []
        for record in self.sequences:
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
        
        pivot_data.plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        ax.set_title('Base Composition Comparison', fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Sequence ID')
        ax.legend(title='Base', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_sequence_lengths(self, ax):
        """Plot sequence length comparison."""
        seq_lengths = [len(str(record.seq)) for record in self.sequences]
        seq_ids = [record.id for record in self.sequences]
        
        bars = ax.bar(range(len(seq_ids)), seq_lengths, 
                     color=plt.cm.plasma(np.linspace(0, 1, len(seq_ids))),
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title('Sequence Length Comparison', fontweight='bold')
        ax.set_ylabel('Length (bp)')
        ax.set_xlabel('Sequence ID')
        ax.set_xticks(range(len(seq_ids)))
        ax.set_xticklabels(seq_ids, rotation=45, ha='right')
        
        # Add value labels
        for bar, length in zip(bars, seq_lengths):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(seq_lengths)*0.01,
                   f'{length:,}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_gc_at_distribution(self, ax):
        """Plot GC vs AT content distribution."""
        gc_at_data = []
        for record in self.sequences:
            seq_str = str(record.seq)
            total = len(seq_str)
            gc_count = seq_str.count('G') + seq_str.count('C')
            at_count = seq_str.count('A') + seq_str.count('T')
            
            gc_at_data.append({
                'Sequence': record.id,
                'GC Content': (gc_count / total) * 100,
                'AT Content': (at_count / total) * 100
            })
        
        df_gc_at = pd.DataFrame(gc_at_data)
        x_pos = range(len(df_gc_at))
        width = 0.6
        
        ax.bar(x_pos, df_gc_at['GC Content'], width, 
               label='GC Content', color='#2E86AB', alpha=0.8)
        ax.bar(x_pos, df_gc_at['AT Content'], width, 
               bottom=df_gc_at['GC Content'], label='AT Content', 
               color='#F18F01', alpha=0.8)
        
        ax.set_title('GC vs AT Content Distribution', fontweight='bold')
        ax.set_ylabel('Content (%)')
        ax.set_xlabel('Sequence ID')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_gc_at['Sequence'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)
    
    def print_summary(self):
        """Print summary of generated files."""
        print("\n🎉 Enhanced analysis complete! Check the 'outputs' folder for all professional results.")
        print("\nGenerated files:")
        print("  📄 dna_analysis.csv - Detailed sequence analysis")
        print("  📊 gc_content_professional.png - Enhanced GC content visualization")
        print("  📈 similarity_analysis_professional.png - Similarity curve and mutation density")
        print("  🎯 comprehensive_base_analysis_professional.png - Complete base composition analysis")
        
        if len(self.sequences) >= 2:
            seq1_str = str(self.sequences[0].seq)
            seq2_str = str(self.sequences[1].seq)
            if self.detect_mutations(seq1_str, seq2_str):
                print("  🧬 mutation_report.csv - Detailed mutation analysis")


def main():
    """Main function to run the DNA analysis pipeline."""
    print("🧬 Professional DNA Sequence Analyzer")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = DNAAnalyzer()
    
    # Load sequences
    if not analyzer.load_sequences():
        return
    
    # Perform analysis
    df_analysis = analyzer.analyze_sequences()
    
    # Generate plots
    analyzer.plot_gc_content(df_analysis)
    analyzer.analyze_mutations_and_similarity()
    analyzer.plot_comprehensive_base_analysis()
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()