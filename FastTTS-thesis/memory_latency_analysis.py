#!/usr/bin/env python
"""
Memory-Latency Analysis for FastTTS vLLM Instances

This script analyzes how allocated memory affects latency for verifier and generator
instances in beam search scenarios. It tests different memory allocations and measures
latency for both prefilling (verifier) and decoding (generator) operations.
"""

import asyncio
import logging
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import multiprocessing as mp

# Set multiprocessing start method
if mp.get_start_method() != 'spawn':
    mp.set_start_method('spawn', force=True)

from fasttts import create_fasttts
from fasttts import FastTTSConfig, SearchConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryLatencyConfig:
    """Configuration for memory-latency analysis."""
    
    # Total memory to allocate (sum of generator and verifier memory)
    total_memory: float = 0.8
    
    # Memory split ratios to test (generator_memory / total_memory)
    gen_memory_alloc: List[float] = None
    
    # Models to test
    generator_model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    verifier_model: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    
    # Beam search parameters
    beam_width: int = 4
    n_beams: int = 16  # Number of beams to maintain
    num_iterations: int = 5
    
    # Test problems
    test_problems: List[str] = None
    
    # Output directory
    output_dir: str = "."

    
    def __post_init__(self):
        if self.gen_memory_alloc is None:
            # Test different splits from 0.1 to 0.9 (10% to 90% for generator)
            self.gen_memory_alloc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        if self.test_problems is None:
            # Use the first sample from AIME2024 dataset
            from datasets import load_dataset
            dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
            self.test_problems = [dataset[0]['problem']]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

class MemoryLatencyAnalyzer:
    """Analyzer for memory-latency relationships in FastTTS."""
    
    def __init__(self, config: MemoryLatencyConfig):
        self.config = config
    
    def _create_fasttts_config(self, gen_memory: float, ver_memory: float) -> FastTTSConfig:
        """Create FastTTS configuration with specific memory allocations."""
        return FastTTSConfig(
            generator_vllm_config={
                "model": self.config.generator_model,
                "gpu_memory_utilization": gen_memory,
            },
            verifier_vllm_config={
                "model": self.config.verifier_model,
                "gpu_memory_utilization": ver_memory,
            },
            offload_enabled=False,
            spec_beam_extension=False,
        )
    
    def _create_search_config(self) -> SearchConfig:
        """Create search configuration for beam search."""
        return SearchConfig(
            approach="beam_search",
            beam_width=self.config.beam_width,
            n=self.config.n_beams,
            num_iterations=self.config.num_iterations,
            temperature=0.8,
        )
    
    def _measure_latency(self, fasttts, problems: List[str], search_config: SearchConfig) -> Dict[str, float]:
        """Measure latency for a single run."""
        try:
            # Initialize models
            fasttts.initialize()
            
            # Measure total time
            start_time = time.time()
            
            # Perform search
            results = fasttts.search(problems, search_config=search_config)
            
            total_time = time.time() - start_time
            
            # Calculate per-token latencies
            effective_num_tokens = results.total_num_tokens
            total_generator_latency = results.total_generator_latency_s
            total_verifier_latency = results.total_verifier_latency_s

            per_token_generator_latency = total_generator_latency / effective_num_tokens if effective_num_tokens > 0 else 0
            per_token_verifier_latency = total_verifier_latency / effective_num_tokens if effective_num_tokens > 0 else 0

            return {
                'total_generator_latency': total_generator_latency,
                'total_verifier_latency': total_verifier_latency,
                'total_time': total_time,
                'effective_num_tokens': effective_num_tokens,
                'num_completions': len(results.completions[0]),
                'per_token_generator_latency': per_token_generator_latency,
                'per_token_verifier_latency': per_token_verifier_latency,
            }
            
        except Exception as e:
            logger.error(f"Error during latency measurement: {e}")
            return None
    
    def test_memory_configuration(self, gen_memory: float, ver_memory: float) -> Dict[str, Any]:
        """Test a specific memory configuration."""
        logger.info(f"Testing configuration: Generator={gen_memory:.2f}, Verifier={ver_memory:.2f}")
        
        config = self._create_fasttts_config(gen_memory, ver_memory)
        search_config = self._create_search_config()
        
        fasttts = create_fasttts(
            generator_vllm_config=config.generator_vllm_config,
            verifier_vllm_config=config.verifier_vllm_config,
            offload_enabled=config.offload_enabled,
            spec_beam_extension=config.spec_beam_extension,
        )
        
        try:
            # Single run as requested
            result = self._measure_latency(fasttts, self.config.test_problems, search_config)
            
            if result:
                return {
                    'gen_memory': gen_memory,
                    'ver_memory': ver_memory,
                    'latencies': result,
                    'num_successful_runs': 1
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error testing configuration: {e}")
            return None
        finally:
            try:
                fasttts.shutdown()
            except:
                pass
    
    def run_analysis(self) -> Dict[str, List[Dict]]:
        """Run the complete memory-latency analysis."""
        logger.info("Starting memory-latency analysis...")
        
        results = []
        
        # Test different memory splits with fixed total memory
        logger.info(f"Testing memory splits with total memory: {self.config.total_memory}")
        
        for gen_memory in self.config.gen_memory_alloc:
            ver_memory = self.config.total_memory - gen_memory
            
            result = self.test_memory_configuration(gen_memory, ver_memory)
            if result:
                results.append(result)
        
        self.results = results
        return results
    
    def create_plots(self, save_path: str = None):
        """Create plots showing memory vs latency relationships."""
        if not self.results:
            logger.error("No results to plot. Run analysis first.")
            return
        
        if save_path is None:
            # Extract model sizes from model names
            gen_size = self.config.generator_model.split('-')[-2] if '-' in self.config.generator_model else "unknown"
            ver_size = self.config.verifier_model.split('-')[-2] if '-' in self.config.verifier_model else "unknown"
            
            # Create descriptive filename in output directory
            save_path = os.path.join(self.config.output_dir, f"memory_latency_n{self.config.n_beams}_gen{gen_size}_ver{ver_size}_total{self.config.total_memory:.1f}.png")
        
        # Create a 2x2 subplot layout for comprehensive analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        gen_memories = [r['gen_memory'] for r in self.results]
        ver_memories = [r['ver_memory'] for r in self.results]
        gen_latencies = [r['latencies']['total_generator_latency'] for r in self.results]
        ver_latencies = [r['latencies']['total_verifier_latency'] for r in self.results]
        gen_per_token_latencies = [r['latencies']['per_token_generator_latency'] for r in self.results]
        ver_per_token_latencies = [r['latencies']['per_token_verifier_latency'] for r in self.results]
        
        # Plot 1: Total Generator Latency
        ax1.plot(gen_memories, gen_latencies, 'bo-', linewidth=2, markersize=8, label='Total Generator')
        ax1.set_xlabel('Generator Memory Utilization')
        ax1.set_ylabel('Latency (seconds)')
        ax1.set_title('Generator Memory vs Total Latency')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add trend line for total generator latency
        if len(gen_memories) > 1:
            z = np.polyfit(gen_memories, gen_latencies, 1)
            p = np.poly1d(z)
            ax1.plot(gen_memories, p(gen_memories), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
            ax1.legend()
        
        # Plot 2: Per-Token Generator Latency
        ax2.plot(gen_memories, gen_per_token_latencies, 'co-', linewidth=2, markersize=8, label='Per-Token Generator')
        ax2.set_xlabel('Generator Memory Utilization')
        ax2.set_ylabel('Latency per Token (seconds)')
        ax2.set_title('Generator Memory vs Per-Token Latency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add trend line for per-token generator latency
        if len(gen_memories) > 1:
            z = np.polyfit(gen_memories, gen_per_token_latencies, 1)
            p = np.poly1d(z)
            ax2.plot(gen_memories, p(gen_memories), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.6f})')
            ax2.legend()
        
        # Plot 3: Total Verifier Latency
        ax3.plot(ver_memories, ver_latencies, 'go-', linewidth=2, markersize=8, label='Total Verifier')
        ax3.set_xlabel('Verifier Memory Utilization')
        ax3.set_ylabel('Latency (seconds)')
        ax3.set_title('Verifier Memory vs Total Latency')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add trend line for total verifier latency
        if len(ver_memories) > 1:
            z = np.polyfit(ver_memories, ver_latencies, 1)
            p = np.poly1d(z)
            ax3.plot(ver_memories, p(ver_memories), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
            ax3.legend()
        
        # Plot 4: Per-Token Verifier Latency
        ax4.plot(ver_memories, ver_per_token_latencies, 'mo-', linewidth=2, markersize=8, label='Per-Token Verifier')
        ax4.set_xlabel('Verifier Memory Utilization')
        ax4.set_ylabel('Latency per Token (seconds)')
        ax4.set_title('Verifier Memory vs Per-Token Latency')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add trend line for per-token verifier latency
        if len(ver_memories) > 1:
            z = np.polyfit(ver_memories, ver_per_token_latencies, 1)
            p = np.poly1d(z)
            ax4.plot(ver_memories, p(ver_memories), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.6f})')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Plots saved to {save_path}")
    
    def save_results(self, save_path: str = None):
        """Save analysis results to JSON file with descriptive name."""
        if save_path is None:
            # Extract model sizes from model names
            gen_size = self.config.generator_model.split('-')[-2] if '-' in self.config.generator_model else "unknown"
            ver_size = self.config.verifier_model.split('-')[-2] if '-' in self.config.verifier_model else "unknown"
            
            # Create descriptive filename in output directory
            save_path = os.path.join(self.config.output_dir, f"memory_latency_n{self.config.n_beams}_gen{gen_size}_ver{ver_size}_total{self.config.total_memory:.1f}.json")
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {save_path}")
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        logger.info("\n" + "="*60)
        logger.info("MEMORY-LATENCY ANALYSIS SUMMARY")
        logger.info("="*60)
        
        if not self.results:
            logger.info("No results to summarize.")
            return
        
        logger.info(f"\nTotal Memory: {self.config.total_memory}")
        logger.info(f"Number of Beams: {self.config.n_beams}")
        logger.info(f"Generator Model: {self.config.generator_model}")
        logger.info(f"Verifier Model: {self.config.verifier_model}")
        
        logger.info("\nResults:")
        logger.info("-" * 80)
        for result in self.results:
            gen_memory = result['gen_memory']
            ver_memory = result['ver_memory']
            gen_latency = result['latencies']['total_generator_latency']
            ver_latency = result['latencies']['total_verifier_latency']
            gen_per_token_latency = result['latencies']['per_token_generator_latency']
            ver_per_token_latency = result['latencies']['per_token_verifier_latency']
            effective_tokens = result['latencies']['effective_num_tokens']
            
            logger.info(f"(Gen: {gen_memory:.2f}, Ver: {ver_memory:.2f}): "
                       f"Gen Latency: {gen_latency:.3f}s, Ver Latency: {ver_latency:.3f}s, "
                       f"Per-Token Gen: {gen_per_token_latency:.6f}s, Per-Token Ver: {ver_per_token_latency:.6f}s, "
                       f"Tokens: {effective_tokens}")
        
        # Find optimal configurations
        if self.results:
            # Best generator latency (total)
            best_gen_total = min(self.results, key=lambda x: x['latencies']['total_generator_latency'])
            logger.info(f"\nBest Generator Configuration (Total Latency):")
            logger.info(f"  Memory: {best_gen_total['gen_memory']:.2f}")
            logger.info(f"  Total Latency: {best_gen_total['latencies']['total_generator_latency']:.3f}s")
            logger.info(f"  Per-Token Latency: {best_gen_total['latencies']['per_token_generator_latency']:.6f}s")
            logger.info(f"  Effective Tokens: {best_gen_total['latencies']['effective_num_tokens']}")
            
            # Best generator latency (per-token)
            best_gen_per_token = min(self.results, key=lambda x: x['latencies']['per_token_generator_latency'])
            logger.info(f"\nBest Generator Configuration (Per-Token Latency):")
            logger.info(f"  Memory: {best_gen_per_token['gen_memory']:.2f}")
            logger.info(f"  Total Latency: {best_gen_per_token['latencies']['total_generator_latency']:.3f}s")
            logger.info(f"  Per-Token Latency: {best_gen_per_token['latencies']['per_token_generator_latency']:.6f}s")
            logger.info(f"  Effective Tokens: {best_gen_per_token['latencies']['effective_num_tokens']}")
            
            # Best verifier latency (total)
            best_ver_total = min(self.results, key=lambda x: x['latencies']['total_verifier_latency'])
            logger.info(f"\nBest Verifier Configuration (Total Latency):")
            logger.info(f"  Memory: {best_ver_total['ver_memory']:.2f}")
            logger.info(f"  Total Latency: {best_ver_total['latencies']['total_verifier_latency']:.3f}s")
            logger.info(f"  Per-Token Latency: {best_ver_total['latencies']['per_token_verifier_latency']:.6f}s")
            logger.info(f"  Effective Tokens: {best_ver_total['latencies']['effective_num_tokens']}")
            
            # Best verifier latency (per-token)
            best_ver_per_token = min(self.results, key=lambda x: x['latencies']['per_token_verifier_latency'])
            logger.info(f"\nBest Verifier Configuration (Per-Token Latency):")
            logger.info(f"  Memory: {best_ver_per_token['ver_memory']:.2f}")
            logger.info(f"  Total Latency: {best_ver_per_token['latencies']['total_verifier_latency']:.3f}s")
            logger.info(f"  Per-Token Latency: {best_ver_per_token['latencies']['per_token_verifier_latency']:.6f}s")
            logger.info(f"  Effective Tokens: {best_ver_per_token['latencies']['effective_num_tokens']}")
            
            # Best combined latency
            best_combined = min(self.results, key=lambda x: x['latencies']['total_generator_latency'] + x['latencies']['total_verifier_latency'])
            logger.info(f"\nBest Combined Configuration:")
            logger.info(f"  Memory (Gen: {best_combined['gen_memory']:.2f}, Ver: {best_combined['ver_memory']:.2f})")
            logger.info(f"  Combined Latency: {best_combined['latencies']['total_generator_latency'] + best_combined['latencies']['total_verifier_latency']:.3f}s")
            logger.info(f"  Generator Per-Token: {best_combined['latencies']['per_token_generator_latency']:.6f}s")
            logger.info(f"  Verifier Per-Token: {best_combined['latencies']['per_token_verifier_latency']:.6f}s")

# 1.5B-7B: 0.19 to 0.33
# 7B-1.5B: 0.68 to 0.75
def main():
    """Main function to run the memory-latency analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Memory-Latency Analysis for FastTTS')
    parser.add_argument('--output-dir', type=str, default='./memory_analysis/7B-1.5B/',
                       help='Output directory for results and plots')
    parser.add_argument('--total-memory', type=float, default=0.90,
                       help='Total memory to allocate')
    parser.add_argument('--gen-memory-alloc', type=float, nargs='+', 
                       default=[ 0.70],
                       help='Generator memory allocation ratios to test')
    parser.add_argument('--n-beams-list', type=int, nargs='+', 
                       default=[128],
                       help='List of n_beams values to test')
    parser.add_argument('--beam-width', type=int, default=4,
                       help='Beam width for search (default: 4)')
    parser.add_argument('--num-iterations', type=int, default=1,
                       help='Number of iterations per test')
    parser.add_argument('--generator-model', type=str, 
                       default="Qwen/Qwen2.5-Math-7B-Instruct",
                       help='Generator model to use')
    parser.add_argument('--verifier-model', type=str,
                       default="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
                       help='Verifier model to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for n_beams in args.n_beams_list:
        # Configuration
        config = MemoryLatencyConfig(
            total_memory=args.total_memory,
            gen_memory_alloc=args.gen_memory_alloc,
            beam_width=args.beam_width,
            n_beams=n_beams,
            num_iterations=args.num_iterations,
            generator_model=args.generator_model,
            verifier_model=args.verifier_model,
            output_dir=args.output_dir
        )
        
        # Create analyzer
        analyzer = MemoryLatencyAnalyzer(config)
        
        # Run analysis
        results = analyzer.run_analysis()
        
        # Create plots
        analyzer.create_plots()
        
        # Save results
        analyzer.save_results()
        
        # Print summary
        analyzer.print_summary()

if __name__ == "__main__":
    main() 