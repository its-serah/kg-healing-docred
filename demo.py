#!/usr/bin/env python3
"""
Demo Script for Knowledge Graph Healing
======================================

Quick demo to test the knowledge graph healing system with a small sample.
"""

import subprocess
import sys
import os

def run_demo():
    """Run a quick demo of the KG healing system"""
    
    print("=" * 60)
    print("KNOWLEDGE GRAPH HEALING - DEMO")
    print("=" * 60)
    
    # Check if Ollama is running
    print("\n1. Checking Ollama status...")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Ollama is running")
            if 'llama3' in result.stdout:
                print("✓ Llama 3 model found")
            else:
                print("⚠ Llama 3 model not found. Run: ollama pull llama3")
                return
        else:
            print("✗ Ollama not responding")
            return
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return
    
    # Check dataset
    print("\n2. Checking dataset...")
    if os.path.exists('Re-DocRED/data/dev_revised.json'):
        print("✓ Re-DocRED dataset found")
    else:
        print("⚠ Re-DocRED dataset not found")
        print("Please run: git clone https://github.com/tonytan48/Re-DocRED.git")
        return
    
    # Run healing on small sample
    print("\n3. Running knowledge graph healing (5 documents)...")
    try:
        cmd = [
            sys.executable, 'kg_healing_enhanced.py',
            '--limit', '5',
            '--output', 'demo_results'
        ]
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("\n✓ Demo completed successfully!")
            print("\nResults saved to 'demo_results/' directory")
            print("\nFiles generated:")
            
            results_dir = 'demo_results'
            if os.path.exists(results_dir):
                for file in os.listdir(results_dir):
                    print(f"  - {file}")
        else:
            print(f"\n✗ Demo failed with exit code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("\n⚠ Demo timed out (>5 minutes)")
    except Exception as e:
        print(f"\n✗ Error running demo: {e}")

if __name__ == "__main__":
    run_demo()
