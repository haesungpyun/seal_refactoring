# import pathlib
# import sys, os
# import importlib
# seal_spec = importlib.util.spec_from_file_location("seal", os.path.abspath(__file__))
# seal = importlib.util.module_from_spec(seal_spec)
# sys.modules["seal"] = seal
# seal_dir = pathlib.Path(__file__).parent
# sys.path.append(str(seal_dir.parent))