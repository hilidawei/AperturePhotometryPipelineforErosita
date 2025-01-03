1. Script Name

erosita_pipeline.py


2. Purpose

This script is designed to process and analyze data from eROSITA X-ray observations. It performs the following tasks:
	•	Load and preprocess eROSITA X-ray data.
	•	Calculate aperture photometric properties for input sources, restricted to the 0.2–2.3 keV energy band and within the r500 radius of the sources.
	•	Convert photometric properties into X-ray luminosities in the 0.5–2.0 keV energy band.
	•	Output processed tables and visualizations for further analysis.


3. Environment

   3.1 Required Dependencies

   The following Python libraries are required to run the script:
	•	numpy
	•	pandas
	•	matplotlib
	•	astropy
	•	regions
	•	xspec

   3.2 Python Version

   The script is compatible with Python 3.10 or later.


4. Data Requirements (***)

    4.1 You need to provide a NumPy array along with a corresponding list of column names for all sources to be measured. The list must include at least the following columns: “ID”, “RA”, “Dec”, “z”, “M500”, and “r500”. If any required column is omitted or if a column name is misspelled, the script is likely to raise an error.

    4.2 The directories of the following files are needed to run the script:
        4.2.1 The complete eRASS1 data archive (e.g., `/path/to/your/ero_archive/`).
        4.2.2 Two specific eFEDS files:
            - `eFEDS_c001_clean_0d2_2d3_Image.fits`
            - `eFEDS_c001_clean_0d2_2d3_ExpMap.fits`
            (e.g., `/path/to/parent_directory/` (the directory containing both files)).
        4.2.3 The eROSITA detected source catalogs for masking
            (e.g., `/path/to/parent_directory/` (the directory containing masking catalogs)).
        4.2.4 On-axis, seven-TMS-combined eROSITA ARF and RMF files
            (e.g., `/path/to/these/two/files/`).


5. Output

You need to provide a directory for all output files.


6. Contact

For questions or issues, please contact:
	•	Author: Dawei
	•	Email: hilidawei@gmail.com, llldawei@stu.xmu.edu.cn