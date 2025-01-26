import h5py
import numpy as np

from pathlib import Path
from gwdatafind import find_urls


glitch_keys = [
    "time", 
    "frequency", 
    "tstart", 
    "tend", 
    "fstart", 
    "fend", 
    "snr", 
    "q", 
    "amplitude", 
    "phase", 
]


def create_lcs(
    ifo: str,
    frametype: str,
    start_time: int,
    end_time: int,
    output_dir: Path,
    urltype: str="file"
):
    """Select time, stateflag >>> get *.gwf file >>> 
    """
    head = "file://localhost"
    empty = ""
    
    files = find_urls(
        site=ifo[0],
        frametype=frametype,
        gpsstart=start_time,
        gpsend=end_time,
        urltype=urltype,
    )
    
    
    output_dir = output_dir / ifo
    output_dir.mkdir(parents=True, exist_ok=True)
    
    f = open(output_dir / "data_file.lcf", "a")
    for file in files:
        f.write(f"{file.replace(head, empty)}\n")
    f.close()



def glitch_merger(
    ifos,
    omicron_path: Path,
    channels,
    output_file=None,
    glitch_keys=glitch_keys
):

    
    for i, ifo in enumerate(ifos):

        gltich_dir = omicron_path / f"{ifo}/trigger_output/merge/{ifo}:{channels[i]}"

        h5_name = {}
        for key in glitch_keys:

            h5_name[key] = []   

        for file in sorted(gltich_dir.glob("*.h5")):

            with h5py.File(file, "r") as h:
                
                for key in glitch_keys:
                    
                    h5_name[key].append(h["triggers"][key])
                    
        for key in glitch_keys:
            h5_name[key] = np.concatenate(h5_name[key])
            
        if output_file is None:
            output_file = omicron_path / "glitch_info.h5"

        with h5py.File(output_file, "a") as g:
            
            g1 = g.create_group(ifo)
            
            for key in glitch_keys:
                g1.create_dataset(key, data=h5_name[key])

    return output_file