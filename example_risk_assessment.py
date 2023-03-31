import shutil
from zipfile import ZipFile
import os
import pandas as pd
from EzBridge import rct
from EzBridge.utils import create_dir

if __name__ == '__main__':
    main_output_directory = os.path.join('example_results', 'assessment')
    create_dir(main_output_directory)
    filenames = os.listdir(os.path.join('example_results', 'analysis'))

    for filename in filenames:
        # Folder names
        input_folder = filename[:-4]
        output_folder = os.path.join(main_output_directory, input_folder)

        # Extract files
        directory_to_extract_to = os.getcwd()
        path_to_zip_file = os.path.join('example_results', 'analysis', input_folder + '.zip')
        with ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        # Perform risk assessment
        risk_obj = rct.msa(input_folder=input_folder, output_folder=output_folder)
        risk_obj.compute_all(plot=1)

        # Remove input folder
        shutil.rmtree(input_folder)
