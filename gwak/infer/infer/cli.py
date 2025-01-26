from export.main import export

from jsonargparse import ArgumentParser, ActionConfigFile


def build_parser():
    
    parser = ArgumentParser(env_prefix="GWAK", default_env=True)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(export)
    
    return parser

def main(args=None):
    
    parser = build_parser()
    args = parser.parse_args()
    args = args.as_dict()
    export(**args)
    
    
if __name__ == "__main__":
    main()

    # # Build Whiten model 
    # ensemble_name = "gwak-stream"
    # try:
    #     # first see if we have an existing
    #     # ensemble with the given name
    #     ensemble = repo.models[ensemble_name]
    # except KeyError:
        
    #     ensemble = repo.add(
    #         ensemble_name, 
    #         platform=qv.Platform.ENSEMBLE
    #     )
        