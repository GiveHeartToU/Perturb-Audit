import argparse
import sys
from perturb_audit.cr_analyzer.runner import CollisionRunner

def main():
    parser = argparse.ArgumentParser(description="CR-Collision-Analyzer Pipeline")
    
    parser.add_argument(
        '--input', '-i', 
        required=True, 
        help="Path to the input molecule_info.h5 file"
    )
    parser.add_argument(
        '--output', '-o', 
        default='results/', 
        help="Directory to save output results"
    )
    parser.add_argument(
        '--config', '-c', 
        default='configs/default_config.yaml', 
        help="Path to the YAML configuration file"
    )
    
    args = parser.parse_args()

    try:
        runner = CollisionRunner(config_path=args.config, output_dir=args.output)
        runner.run_pipeline(h5_path=args.input)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()