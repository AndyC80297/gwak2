import torch
from pathlib import Path

import hermes.quiver as qv

from export.utils import scale_model, add_whiten_streamer

def export(
    project: Path,
    output_dir: Path,
    triton_dir: Path,
    clean: bool,
    batch_size: int, 
    kernel_size: int, 
    num_ifos: int, 
    gwak_instances: int, 
    # psd_length: int,
    # highpass: int,
    # fftlength: int,
    # inference_sampling_rate: int,
    # preproc_instances: int,
    # streams_per_gpu: int,
    platform: qv.Platform = qv.Platform.ONNX,
    **kwargs,
):
    
    weights = output_dir / project / "model_JIT.pt"
    
    with open(weights, "rb") as f:
        graph = torch.jit.load(f)

    graph.eval()

    triton_dir.mkdir(parents=True, exist_ok=True)
    repo = qv.ModelRepository(triton_dir, clean=clean)

    try:
        gwak = repo.models[f"gwak-{project}"]
    except KeyError:
        gwak = repo.add(f"gwak-{project}", qv.Platform.TENSORRT)

    if gwak_instances is not None:
        scale_model(gwak, gwak_instances)

    input_shape = (batch_size, kernel_size, num_ifos)

    kwargs = {}
    if platform == qv.Platform.ONNX:
        kwargs["opset_version"] = 13

        # turn off graph optimization because of this error
        # https://github.com/triton-inference-server/server/issues/3418
        gwak.config.optimization.graph.level = -1
    elif platform == qv.Platform.TENSORRT:
        kwargs["use_fp16"] = False
    
    gwak.export_version(
        graph, 
        input_shapes={"whitened": input_shape}, 
        output_names=["classifier"],
        **kwargs,
    )

    # Build Whiten model 
    ensemble_name = "gwak-stream"
    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        
        ensemble = repo.add(
            ensemble_name, 
            platform=qv.Platform.ENSEMBLE
        )
        