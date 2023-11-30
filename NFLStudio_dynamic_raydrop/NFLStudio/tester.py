from NFLStudio.nflmodel import *
from NFLStudio.waymo_dynamic import *
from nerfstudio.pipelines.base_pipeline import *
from NFLStudio.nflpipeline import *
import os
context_name = '1005081002024129653_5313_150_5333_150'
# context_name = '1083056852838271990_4080_000_4100_000_shifted'
# context_name = '1083056852838271990_4080_000_4100_000_replace'
# context_name = '10963653239323173269_1924_000_1944_000'
# context_name = '11967272535264406807_580_000_600_000'
# context_name = '13271285919570645382_5320_000_5340_000_singleshifted'
# context_name = '13271285919570645382_5320_000_5340_000'
# context_name = '10072140764565668044_4060_000_4080_000_shifted'
# context_name = '10588771936253546636_2300_000_2320_000'
# context_name = '10500357041547037089_1474_800_1494_800_shifted'
# operation = '_SHIFT'
# operation = '_LIFT'
# operation = '_TILT'
# operation = '_DOWNSAMPLE'
# operation = '_UPSAMPLE'
# operation = '_NONE'
# operation = '_REMOVAL_'
# operation = '_TRAJSHIFT'
operation = '_INSERT'
def _load_checkpoint(pipeline, load_dir, load_step) -> None:
    """Helper function to load pipeline and optimizer from prespecified checkpoint"""
    load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    # load the checkpoints for pipeline, optimizers, and gradient scalar
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
if __name__ == '__main__':
    pipeline = NFLPipeline(config=NFLPipelineConfig(), device="cuda", world_size=1)
    # load_dir = f'/scratch/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/outputs/6_replaced_with_9/'
    load_dir = f'/scratch/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/outputs/5_replaced_with_9'
    load_dir = Path(load_dir)
    for load_step in sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1:]:
        print(load_step)
        _load_checkpoint(pipeline, load_dir, load_step)
        # pipeline.get_test_loss_dict(1)
        pipeline.model.nfl_field.set_cos_anneal_ratio(min(1.0, load_step/40000))
        # pipeline.model.nfl_field.encoder_sdf.update_step(0,load_step)
        # for i in range(len(pipeline.model.vehicle_fields)):
        #     pipeline.model.vehicle_fields[i].encoder_sdf.update_step(0,load_step)
        # pipeline.get_numbers(1)

        pipeline.get_pcd(context_name+operation+"_FULL")
        # pipeline.get_dynamic_recall()
