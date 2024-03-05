from NFLStudio.nflmodel import *
from NFLStudio.waymo_dynamic import *
from nerfstudio.pipelines.base_pipeline import *
from NFLStudio.nflpipeline import *
import os
import argparse
# context_name = '1005081002024129653_5313_150_5333_150'
# context_name = '1083056852838271990_4080_000_4100_000'
# context_name = '13271285919570645382_5320_000_5340_000'
# context_name = '10072140764565668044_4060_000_4080_000'
# context_name = '10500357041547037089_1474_800_1494_800'
# operation = '_SHIFT'
# operation = '_LIFT'
# operation = '_TILT'
# operation = '_DOWNSAMPLE'
# operation = '_UPSAMPLE'
# operation = '_NONE'
# operation = '_REMOVAL_'
# operation = '_TRAJSHIFT'
# operation = '_INSERT'

def _load_checkpoint(pipeline, load_dir, load_step) -> None:
    """Helper function to load pipeline and optimizer from prespecified checkpoint"""
    load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    # load the checkpoints for pipeline, optimizers, and gradient scalar
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tester')
    parser.add_argument('--context_name', type=str,
                    help='Context name of the to test driving scenario')
    parser.add_argument('--model_dir', type=str,
                    help='Path to directory that stores the checkpoint, we use the last checkpoint')
    args = parser.parse_args()
    context_name = args.context_name
    load_dir = args.model_dir
    load_dir = Path(load_dir)
    pipeline = NFLPipeline(config=NFLPipelineConfig(datamanager=NFLDataManagerConfig(dataparser_config=NFLDataParserConfig(context_name=context_name))), device="cuda", world_size=1)
    for load_step in sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1:]:
        print(load_step)
        _load_checkpoint(pipeline, load_dir, load_step)
        pipeline.model.nfl_field.set_cos_anneal_ratio(min(1.0, load_step/40000))
        
        pipeline.get_numbers() # get numbers of the LiDARs
        # pipeline.get_pcd(context_name)# get pcd to display
