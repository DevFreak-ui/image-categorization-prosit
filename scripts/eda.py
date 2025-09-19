#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for CINIC-10 Dataset
Analyzes dataset structure, class distribution, and image characteristics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from collections import Counter
import json
from datetime import datetime

def analyze_dataset_structure(data_root='./dataset'):
    """Analyze the basic structure of the CINIC-10 dataset"""
    print("=" * 60)
    print("CINIC-10 DATASET STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(data_root):
        print(f"Error: Dataset directory not found at {data_root}")
        return None
    
    # Analyze directory structure
    splits = ['train', 'valid', 'test']
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    structure_info = {}
    
    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            print(f"{split} directory not found")
            continue
            
        print(f"\n📁 {split.upper()} SET:")
        print("-" * 30)
        
        split_info = {}
        total_images = 0
        
        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                # Count images in this class
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                image_count = len(image_files)
                total_images += image_count
                split_info[class_name] = image_count
                print(f"  {class_name:12}: {image_count:6,} images")
            else:
                print(f"  {class_name:12}: NOT FOUND")
                split_info[class_name] = 0
        
        print(f"  {'TOTAL':12}: {total_images:6,} images")
        structure_info[split] = split_info
    
    return structure_info

def analyze_class_distribution(structure_info):
    """Analyze and visualize class distribution across splits"""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Create DataFrame for analysis
    df_data = []
    for split, classes in structure_info.items():
        for class_name, count in classes.items():
            df_data.append({
                'Split': split,
                'Class': class_name,
                'Count': count
            })
    
    df = pd.DataFrame(df_data)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    summary = df.groupby('Split')['Count'].agg(['sum', 'mean', 'std', 'min', 'max'])
    print(summary)
    
    # Class distribution across splits
    print("\nCLASS DISTRIBUTION ACROSS SPLITS:")
    print("-" * 50)
    pivot_df = df.pivot(index='Class', columns='Split', values='Count')
    print(pivot_df)
    
    # Calculate percentages
    print("\nPERCENTAGE DISTRIBUTION:")
    print("-" * 40)
    for split in ['train', 'valid', 'test']:
        if split in pivot_df.columns:
            total = pivot_df[split].sum()
            print(f"\n{split.upper()} SET:")
            for class_name in pivot_df.index:
                count = pivot_df.loc[class_name, split]
                percentage = (count / total) * 100
                print(f"  {class_name:12}: {percentage:5.1f}%")
    
    return df, pivot_df

def visualize_class_distribution(df, pivot_df, save_plots=True):
    """Create 3 comprehensive distribution charts with light, subtle colors"""
    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Set up beautiful plotting style with light colors
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Distribution Analysis on CINIC-10', 
                 fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
    
    # Define light, subtle color palettes
    light_colors = ['#E8F4FD', '#D1E7DD', '#F8D7DA', '#FFF3CD', '#D4EDDA', 
                   '#F5C6CB', '#E2E3E5', '#D1ECF1', '#FDEBD0', '#E8EAF6']
    split_colors = ['#B8D4E3', '#A8D5BA', '#F4A6B7']  # Light blue, light green, light pink
    gradient_colors = ['#F0F8FF', '#E6F3FF', '#D6EBFF', '#C6E2FF', '#B6D9FF']
    
    # 1. Class Balance Analysis - Stacked bar chart showing distribution across splits
    ax1 = axes[0]
    pivot_df.plot(kind='bar', ax=ax1, width=0.8, color=split_colors, 
                 edgecolor='white', linewidth=1, alpha=0.8)
    ax1.set_title('Class Distribution Across Dataset Splits', 
                 fontsize=14, fontweight='bold', pad=20, color='#34495e')
    ax1.set_xlabel('Image Classes', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.legend(title='Dataset Split', title_fontsize=11, fontsize=10, 
              frameon=True, fancybox=True, shadow=True, facecolor='white')
    ax1.tick_params(axis='x', rotation=45, labelsize=10, colors='#2c3e50')
    ax1.tick_params(axis='y', labelsize=10, colors='#2c3e50')
    ax1.grid(True, alpha=0.2, linestyle='-', color='#bdc3c7')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#bdc3c7')
    ax1.spines['bottom'].set_color('#bdc3c7')
    
    # Add subtle background
    ax1.set_facecolor('#FAFAFA')
    
    # 2. Dataset Structure - Donut chart with detailed breakdown
    ax2 = axes[1]
    total_per_split = pivot_df.sum(axis=0)
    # Create custom labels with counts
    labels_with_counts = [f'{split}\n({count:,} images)' for split, count in total_per_split.items()]
    
    wedges, texts, autotexts = ax2.pie(total_per_split.values, labels=labels_with_counts, 
                                      colors=split_colors, autopct='%1.1f%%', startangle=90,
                                      pctdistance=0.75, textprops={'fontsize': 10, 'fontweight': 'bold'},
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    # Create donut effect
    centre_circle = plt.Circle((0,0), 0.60, fc='white', edgecolor='#bdc3c7', linewidth=1)
    ax2.add_artist(centre_circle)
    ax2.set_title('Dataset Structure & Split Proportions', 
                 fontsize=14, fontweight='bold', pad=20, color='#34495e')
    
    # Add comprehensive info in center
    total_images = total_per_split.sum()
    avg_per_class = total_images / len(pivot_df)
    ax2.text(0, 0, f'Total Dataset\n{total_images:,} Images\n\nAvg per Class\n{avg_per_class:,.0f} Images', 
             ha='center', va='center', fontsize=11, fontweight='bold', 
             color='#2c3e50', linespacing=1.2)
    
    # 3. Class Balance Analysis - Horizontal bar with balance indicators
    ax3 = axes[2]
    total_per_class = pivot_df.sum(axis=1).sort_values(ascending=True)
    
    # Create gradient colors based on balance
    min_count = total_per_class.min()
    max_count = total_per_class.max()
    balance_ratio = min_count / max_count
    
    # Color bars based on how balanced they are
    bar_colors = []
    for count in total_per_class.values:
        # More balanced classes get lighter colors, imbalanced get slightly darker
        if count == min_count:
            bar_colors.append('#E8F4FD')  # Lightest for smallest
        elif count == max_count:
            bar_colors.append('#B8D4E3')  # Darkest for largest
        else:
            # Interpolate between light and slightly darker
            ratio = (count - min_count) / (max_count - min_count)
            bar_colors.append(f'#{int(232 + ratio * 40):02x}{int(244 + ratio * 20):02x}{int(253 - ratio * 20):02x}')
    
    bars = ax3.barh(range(len(total_per_class)), total_per_class.values, 
                   color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax3.set_yticks(range(len(total_per_class)))
    ax3.set_yticklabels(total_per_class.index, fontsize=11, color='#2c3e50')
    ax3.set_xlabel('Total Images per Class', fontsize=12, fontweight='bold', color='#2c3e50')
    ax3.set_title(f'Class Balance Analysis\n(Balance Ratio: {balance_ratio:.2f})', 
                 fontsize=14, fontweight='bold', pad=20, color='#34495e')
    ax3.tick_params(axis='x', labelsize=10, colors='#2c3e50')
    ax3.grid(True, alpha=0.2, axis='x', linestyle='-', color='#bdc3c7')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#bdc3c7')
    ax3.spines['bottom'].set_color('#bdc3c7')
    
    # Add value labels with balance indicators
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # Add balance indicator
        if width == min_count:
            indicator = " (min)"
        elif width == max_count:
            indicator = " (max)"
        else:
            indicator = ""
        
        ax3.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}{indicator}', ha='left', va='center', 
                fontsize=9, fontweight='bold', color='#2c3e50')
    
    # Add subtle background
    ax3.set_facecolor('#FAFAFA')
    
    # Add balance quality indicator
    if balance_ratio > 0.9:
        balance_quality = "Excellent"
        balance_color = "#27AE60"
    elif balance_ratio > 0.8:
        balance_quality = "Good"
        balance_color = "#F39C12"
    else:
        balance_quality = "Fair"
        balance_color = "#E74C3C"
    
    ax3.text(0.02, 0.98, f'Balance Quality: {balance_quality}', 
             transform=ax3.transAxes, fontsize=10, fontweight='bold', 
             color=balance_color, bbox=dict(boxstyle="round,pad=0.3", 
             facecolor='white', alpha=0.8, edgecolor=balance_color))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('cinic10_comprehensive_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("Comprehensive distribution analysis saved as: cinic10_comprehensive_analysis.png")
    
    plt.show()
    
    return fig

def analyze_image_characteristics(data_root='./dataset', sample_size=100):
    """Analyze image characteristics (size, format, etc.)"""
    print("\n" + "=" * 60)
    print("IMAGE CHARACTERISTICS ANALYSIS")
    print("=" * 60)
    
    # Sample images from each split
    splits = ['train', 'valid', 'test']
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    image_info = {
        'widths': [],
        'heights': [],
        'formats': [],
        'file_sizes': [],
        'split': [],
        'class': []
    }
    
    print(f"Analyzing {sample_size} random images from each split...")
    
    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"\n  Processing {split} set...")
        
        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            # Get random sample of images
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                continue
                
            # Sample images
            sample_files = np.random.choice(image_files, 
                                          min(sample_size, len(image_files)), 
                                          replace=False)
            
            for img_file in sample_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_info['widths'].append(width)
                        image_info['heights'].append(height)
                        image_info['formats'].append(img.format)
                        image_info['file_sizes'].append(os.path.getsize(img_path))
                        image_info['split'].append(split)
                        image_info['class'].append(class_name)
                except Exception as e:
                    print(f"    Error processing {img_file}: {e}")
    
    # Convert to DataFrame
    df_images = pd.DataFrame(image_info)
    
    if df_images.empty:
        print("No images found for analysis")
        return None
    
    # Print statistics
    print(f"\nIMAGE STATISTICS (from {len(df_images)} samples):")
    print("-" * 50)
    print(f"Width:  {df_images['widths'].min():3d} - {df_images['widths'].max():3d} (mean: {df_images['widths'].mean():.1f})")
    print(f"Height: {df_images['heights'].min():3d} - {df_images['heights'].max():3d} (mean: {df_images['heights'].mean():.1f})")
    print(f"File sizes: {df_images['file_sizes'].min():,} - {df_images['file_sizes'].max():,} bytes")
    
    # Format distribution
    print(f"\n IMAGE FORMATS:")
    format_counts = df_images['formats'].value_counts()
    for fmt, count in format_counts.items():
        percentage = (count / len(df_images)) * 100
        print(f"  {fmt}: {count:4d} ({percentage:5.1f}%)")
    
    return df_images

def visualize_image_characteristics(df_images, save_plots=True):
    """Create beautiful visualizations for image characteristics"""
    if df_images is None or df_images.empty:
        print(" No image data to visualize")
        return
    
    print("\n Creating beautiful image characteristics visualizations...")
    
    # Set up beautiful plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(' CINIC-10 Image Characteristics Beautiful Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Beautiful hexbin plot for dimensions
    ax1 = axes[0, 0]
    hb = ax1.hexbin(df_images['widths'], df_images['heights'], gridsize=20, cmap='Blues', alpha=0.8)
    ax1.set_xlabel('Width (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Height (pixels)', fontsize=12, fontweight='bold')
    ax1.set_title(' mage Dimensions Density', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.colorbar(hb, ax=ax1, label='Count')
    
    # 2. Beautiful KDE plot for file size distribution
    ax2 = axes[0, 1]
    sns.kdeplot(data=df_images, x='file_sizes', ax=ax2, fill=True, alpha=0.7, color='#FF6B6B')
    ax2.set_xlabel('File Size (bytes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('File Size Distribution (KDE)', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Beautiful donut chart for image formats
    ax3 = axes[0, 2]
    format_counts = df_images['formats'].value_counts()
    colors_format = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    wedges, texts, autotexts = ax3.pie(format_counts.values, labels=format_counts.index, 
                                      colors=colors_format[:len(format_counts)], 
                                      autopct='%1.1f%%', startangle=90,
                                      pctdistance=0.85, textprops={'fontsize': 10, 'fontweight': 'bold'})
    # Create donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax3.add_artist(centre_circle)
    ax3.set_title('Image Format Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # 4. Beautiful box plot for dimensions by class
    ax4 = axes[1, 0]
    # Create a combined dimension metric
    df_images['total_pixels'] = df_images['widths'] * df_images['heights']
    sns.boxplot(data=df_images, x='class', y='total_pixels', ax=ax4, palette='viridis')
    ax4.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Total Pixels', fontsize=12, fontweight='bold')
    ax4.set_title('Image Size Distribution by Class', fontsize=14, fontweight='bold', pad=20)
    ax4.tick_params(axis='x', rotation=45, labelsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Beautiful scatter plot with class colors
    ax5 = axes[1, 1]
    class_dims = df_images.groupby('class').agg({'widths': 'mean', 'heights': 'mean'})
    colors_scatter = plt.cm.tab10(np.linspace(0, 1, len(class_dims)))
    
    scatter = ax5.scatter(class_dims['widths'], class_dims['heights'], 
                         s=200, c=colors_scatter, alpha=0.8, edgecolors='white', linewidth=2)
    
    for i, class_name in enumerate(class_dims.index):
        ax5.annotate(class_name, (class_dims.iloc[i]['widths'], class_dims.iloc[i]['heights']), 
                    xytext=(8, 8), textcoords='offset points', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax5.set_xlabel('Average Width (pixels)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Average Height (pixels)', fontsize=12, fontweight='bold')
    ax5.set_title('Average Dimensions by Class', fontsize=14, fontweight='bold', pad=20)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # 6. Beautiful violin plot for file sizes by split
    ax6 = axes[1, 2]
    sns.violinplot(data=df_images, x='split', y='file_sizes', ax=ax6, palette=['#FF6B6B', '#4ECDC4', '#45B7D1'], inner='box')
    ax6.set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
    ax6.set_ylabel('File Size (bytes)', fontsize=12, fontweight='bold')
    ax6.set_title('File Size Distribution by Split', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('cinic10_image_characteristics.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Beautiful image characteristics plot saved as: cinic10_image_characteristics.png")
    
    plt.show()
    
    return fig

def save_eda_report(structure_info, df, pivot_df, df_images, save_dir='eda_results'):
    """Save EDA results to files"""
    print("\n" + "=" * 60)
    print("SAVING EDA REPORT")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save structure info as JSON
    structure_file = os.path.join(save_dir, f'cinic10_structure_{timestamp}.json')
    with open(structure_file, 'w') as f:
        json.dump(structure_info, f, indent=2)
    print(f" Structure info saved to: {structure_file}")
    
    # Save class distribution as CSV
    if df is not None:
        distribution_file = os.path.join(save_dir, f'cinic10_distribution_{timestamp}.csv')
        df.to_csv(distribution_file, index=False)
        print(f"Class distribution saved to: {distribution_file}")
    
    # Save image characteristics as CSV
    if df_images is not None:
        characteristics_file = os.path.join(save_dir, f'cinic10_characteristics_{timestamp}.csv')
        df_images.to_csv(characteristics_file, index=False)
        print(f" Image characteristics saved to: {characteristics_file}")
    
    # Create summary report
    summary_file = os.path.join(save_dir, f'cinic10_eda_summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("CINIC-10 Dataset EDA Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Dataset Structure:\n")
        f.write("-" * 20 + "\n")
        for split, classes in structure_info.items():
            f.write(f"{split.upper()}:\n")
            for class_name, count in classes.items():
                f.write(f"  {class_name}: {count:,} images\n")
            f.write(f"  Total: {sum(classes.values()):,} images\n\n")
        
        if df_images is not None:
            f.write("Image Characteristics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Sample size: {len(df_images)} images\n")
            f.write(f"Width range: {df_images['widths'].min()} - {df_images['widths'].max()} pixels\n")
            f.write(f"Height range: {df_images['heights'].min()} - {df_images['heights'].max()} pixels\n")
            f.write(f"File size range: {df_images['file_sizes'].min():,} - {df_images['file_sizes'].max():,} bytes\n")
            f.write(f"Formats: {', '.join(df_images['formats'].unique())}\n")
    
    print(f"Summary report saved to: {summary_file}")

def main():
    """Main EDA function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis for CINIC-10 Dataset')
    parser.add_argument('--data_root', type=str, default='./dataset', 
                       help='Path to CINIC-10 dataset')
    parser.add_argument('--sample_size', type=int, default=100, 
                       help='Number of images to sample for characteristics analysis')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save plots as PNG files')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save any files')
    
    args = parser.parse_args()
    
    print("CINIC-10 Dataset Exploratory Data Analysis")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Analyze dataset structure
    structure_info = analyze_dataset_structure(args.data_root)
    if structure_info is None:
        return
    
    # 2. Analyze class distribution
    df, pivot_df = analyze_class_distribution(structure_info)
    
    # 3. Visualize class distribution
    if not args.no_save:
        visualize_class_distribution(df, pivot_df, save_plots=args.save_plots)
    
    # 4. Save results
    if not args.no_save:
        save_eda_report(structure_info, df, pivot_df, None)
    
    print("\nEDA Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
