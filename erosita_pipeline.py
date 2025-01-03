#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@time: 2024/12/30
@author: Dawei Li
@contact: llldawei@stu.xmu.edu.cn
@description: Pipeline for processing eROSITA X-ray data.
"""

# === Imports ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from regions import CircleSkyRegion, Regions
from xspec import AllData, AllModels, Model, Xset, FakeitSettings
import warnings

# === Global Configuration ===
warnings.filterwarnings('ignore', category=FITSFixedWarning)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

# === Utility Functions ===

def numpy_to_pandas(data, columns=None):
    """
    Convert a 2D numpy array to a Pandas DataFrame.
    """
    # Default column names if not provided
    if columns is None:
        columns = ['ID', 'RA', 'Dec', 'z', 'M500', 'r500']

    if data.shape[1] != len(columns):
        raise ValueError(f"Input data must have {len(columns)} columns, got {data.shape[1]}.")

    data_df = pd.DataFrame(data, columns=columns)
    data_df = data_df.astype({
        'ID': int, 'RA': float, 'Dec': float, 'z': float, 'M500': float, 'r500': float
    })

    return data_df


def find_eFEDS_files(data_df, dir, output_dir, cutout_size_deg=4.0):
    """
    Find corresponding eFEDS image and exposure map for each source based on its RA and Dec,
    then cut out a region centered on the source with a specified diameter.

    Parameters:
    ----------
    data_df : pd.DataFrame
        DataFrame with source information, including 'ID', 'RA', 'Dec', 'z', 'M500', 'r500'.
    dir : str
        Directory of the eFEDS 0.2-2.3 keV images and exposure maps.
    output_dir : str
        Directory to save cutout images and exposure maps.
    cutout_size_deg : float, optional
        Diameter of the cutout region in degrees (default is 4.0 degrees).

    Returns:
    -------
    pd.DataFrame
        Updated DataFrame with new 'image_path' and 'expmap_path' columns.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the paths to the full eFEDS image and exposure map
    full_image_path = os.path.join(dir, 'eFEDS_c001_clean_0d2_2d3_Image.fits')
    full_expmap_path = os.path.join(dir, 'eFEDS_c001_clean_0d2_2d3_ExpMap.fits')

    if not os.path.isfile(full_image_path) or not os.path.isfile(full_expmap_path):
        raise FileNotFoundError("eFEDS image or exposure map not found in the specified directory.")

    # Initialize lists to store the paths of the cutout files
    cutout_image_paths = []
    cutout_expmap_paths = []

    # Load the full image and exposure map
    with fits.open(full_image_path) as img_hdul, fits.open(full_expmap_path) as exp_hdul:
        img_data = img_hdul[0].data
        img_header = img_hdul[0].header
        wcs = WCS(img_header)

        exp_data = exp_hdul[0].data
        exp_header = exp_hdul[0].header

        # Loop over each source in the DataFrame
    for _, row in data_df.iterrows():
        ra, dec = row['RA'], row['Dec']
        source_coord = SkyCoord(ra, dec, unit='deg', frame='icrs')

        # Calculate cutout size in pixels
        cutout_size = (cutout_size_deg * 3600) / 4 # Convert degrees to pixels
        print(f"Cutout size: {cutout_size} pixels")

        # Create cutouts for the image and exposure map
        try:
            cutout_image = Cutout2D(data=img_data, position=source_coord, size=cutout_size, wcs=wcs)
            cutout_expmap = Cutout2D(data=exp_data, position=source_coord, size=cutout_size, wcs=wcs)


            # Generate new file paths
            cutout_image_path = os.path.join(output_dir, f"cutout_eFEDS_image_id{row['ID']}.fits")
            cutout_expmap_path = os.path.join(output_dir, f"cutout_eFEDS_expmap_id{row['ID']}.fits")

            # Save cutouts with updated headers
            fits.writeto(cutout_image_path, cutout_image.data, cutout_image.wcs.to_header(), overwrite=True)
            fits.writeto(cutout_expmap_path, cutout_expmap.data, cutout_expmap.wcs.to_header(), overwrite=True)

            # Append paths to lists
            cutout_image_paths.append(cutout_image_path)
            cutout_expmap_paths.append(cutout_expmap_path)

        except Exception as e:
            print(f"Error creating cutouts for source ID {row['ID']}: {e}")
            cutout_image_paths.append(None)
            cutout_expmap_paths.append(None)

    # Add cutout file paths to the DataFrame
    data_df['image_path'] = cutout_image_paths
    data_df['expmap_path'] = cutout_expmap_paths
    matched_files_df = data_df.copy()

    return matched_files_df


def process_sources_with_mask_eFEDS(matched_files_df, mask_catalog_dir, output_dir, src_radius=80, verbose=True):
    """
    Generate and directly apply mask regions to image and exposure map files in memory,
    and return a new DataFrame with paths to the masked files.

    Parameters:
    ----------
    matched_files_df : pd.DataFrame
        DataFrame containing source information, including:
        'id', 'RA', 'Dec', 'z', 'M500', 'r500', 'image_path', 'expmap_path'.
    mask_catalog_dir : str
        Directory containing mask catalog files ('eRASS1_Main.v1.1.fits' and 'eRASS1_Supp.v1.1.fits').
    output_dir : str
        Directory to save masked image and exposure map files.
    src_radius : float, optional
        Mask radius around sources in arcseconds (default is 100).
    verbose : bool, optional
        If True, print progress and debugging information. Default is True.

    Returns:
    -------
    pd.DataFrame
        New DataFrame with added columns for masked image and exposure map paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load mask catalogs
    mask1 = Table.read(os.path.join(mask_catalog_dir, 'eFEDS_c001_main_V7.4.fits')).as_array()
    mask2 = Table.read(os.path.join(mask_catalog_dir, 'eFEDS_c001_supp_V7.4.fits')).as_array()
    mask = np.concatenate((mask1, mask2))
    if mask.shape[0] != mask1.shape[0] + mask2.shape[0]:
        raise ValueError("Combined mask shape does not match expected shape.")

    ra_column, dec_column = mask['RA'], mask['DEC']
    half_size = 2

    # Initialize lists to store paths of masked files
    masked_image_paths = []
    masked_expmap_paths = []

    for index, row in matched_files_df.iterrows():
        source_id = row['ID']
        ra, dec = row['RA'], row['Dec']
        image_path = row['image_path']
        exp_path = row['expmap_path']

        # Generate mask directly in memory
        mask_condition = (
            (ra_column >= ra - half_size) & (ra_column <= ra + half_size) &
            (dec_column >= dec - half_size) & (dec_column <= dec + half_size)
        )
        filtered_mask = mask[mask_condition]

        radius_deg = src_radius / 3600
        distances = np.sqrt((filtered_mask['RA'] - ra)**2 + (filtered_mask['DEC'] - dec)**2)
        final_mask = filtered_mask[distances > radius_deg]

        try:
            # Open image and exposure map files
            with fits.open(image_path) as hdul:
                header = hdul[0].header
                wcs = WCS(header)
                image_data = CCDData(hdul[0].data, unit="adu", header=header)

            with fits.open(exp_path) as hdul:
                exp_data = CCDData(hdul[0].data, unit="adu", header=hdul[0].header)

            ny, nx = image_data.shape
            mask_array = np.ones((ny, nx), dtype=bool)

            for ra_mask, dec_mask, ext_mask in zip(final_mask['RA'], final_mask['DEC'], final_mask['EXT']):
                if ext_mask == 0:
                    mask_radius_deg = 40 / 3600
                elif ext_mask < 60:
                    mask_radius_deg = (ext_mask + 40) / 3600
                else:
                    mask_radius_deg = (ext_mask * 3) / 3600

                region = CircleSkyRegion(
                    center=SkyCoord(ra=ra_mask * u.deg, dec=dec_mask * u.deg, frame='icrs'),
                    radius=mask_radius_deg * u.deg
                )
                region_pix = region.to_pixel(wcs)
                mask_cutout = region_pix.to_mask(mode='center').to_image((ny, nx))
                if mask_cutout is not None:
                    mask_array &= np.logical_not(mask_cutout)

            # Apply mask
            masked_image = image_data.data * mask_array
            masked_exp = exp_data.data * mask_array

            # Save masked results
            output_image_path = os.path.join(output_dir, f"masked_eFEDS_image_id{source_id}.fits")
            output_exp_path = os.path.join(output_dir, f"masked_eFEDS_expmap_id{source_id}.fits")
            fits.writeto(output_image_path, masked_image, image_data.header, overwrite=True)
            fits.writeto(output_exp_path, masked_exp, exp_data.header, overwrite=True)

            if verbose:
                print(f"[{source_id}] Masked image saved: {output_image_path}")
                print(f"[{source_id}] Masked exposure map saved: {output_exp_path}")

            # Append paths to lists
            masked_image_paths.append(output_image_path)
            masked_expmap_paths.append(output_exp_path)

        except Exception as e:
            print(f"[{source_id}] Error processing files: {e}")
            # Append None for failed cases
            masked_image_paths.append(None)
            masked_expmap_paths.append(None)
            continue

    # Create a new DataFrame with additional columns
    updated_matched_files_df = matched_files_df.copy()
    updated_matched_files_df['masked_image_path'] = masked_image_paths
    updated_matched_files_df['masked_expmap_path'] = masked_expmap_paths

    if verbose:
        print("All sources processed and DataFrame updated.")

    return updated_matched_files_df   



def find_eRASS_files(data_df, dir):
    """
    Find corresponding eROSITA image and exposure map for each source based on its RA and Dec.
    
    Parameters:
    data_df: Pandas DataFrame with source information, including 'id', 'RA', 'Dec', 'z', 'M500', 'r500'.
    dir: the directory of the eRASS1 data archive.
    
    Returns:
    A Pandas DataFrame with source information, including 'id', 'RA', 'Dec', 'z', 'M500', 'r500', 
    'eROSITA_image', 'eROSITA_expmap'.
    """
    
    def is_within_wcs(ra, dec, image_file):
        """
        Check if the given RA and Dec fall within the WCS bounds of the image file.
        """
        with fits.open(image_file) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            naxis1 = header['NAXIS1']
            naxis2 = header['NAXIS2']
            
            # Define corners of the image in pixel coordinates
            corners = np.array([[0, 0], [naxis1, 0], [0, naxis2], [naxis1, naxis2]])
            world_corners = wcs.all_pix2world(corners, 1)
            ra_vals = world_corners[:, 0]
            dec_vals = world_corners[:, 1]
            
            # RA and Dec bounds
            ra_min, ra_max = ra_vals.min(), ra_vals.max()
            dec_min, dec_max = dec_vals.min(), dec_vals.max()
            
            # Check if RA and Dec fall within the bounds
            return ra_min <= ra <= ra_max and dec_min <= dec <= dec_max
    
    # Prepare the output list
    result = []


    for _, row in data_df.iterrows():
        # 将所有行数据转为字典形式
        row_data = row.to_dict()
        found_image = False

        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith("024_Image_c010.fits.gz") and not file.startswith("."):
                    image_path = os.path.join(root, file)
                    if is_within_wcs(row['RA'], row['Dec'], image_path):  # 假设 is_within_wcs 使用 RA 和 Dec
                        # 替换路径生成 expmap_path
                        expmap_path = image_path.replace('EXP_010', 'DET_010').replace('Image', 'ExposureMap')
                        
                        # 添加新的路径信息到当前行数据
                        row_data['image_path'] = image_path
                        row_data['expmap_path'] = expmap_path
                        result.append(row_data)
                        
                        found_image = True
                        break
            if found_image:
                break
    
    # Convert result to Pandas DataFrame
    matched_files_df = pd.DataFrame(result)
    return matched_files_df


def process_sources_with_mask_eRASS(matched_files_df, mask_catalog_dir, output_dir, src_radius=80, verbose=True):
    """
    Generate and directly apply mask regions to image and exposure map files in memory,
    and return a new DataFrame with paths to the masked files.

    Parameters:
    ----------
    matched_files_df : pd.DataFrame
        DataFrame containing source information, including:
        'id', 'RA', 'Dec', 'z', 'M500', 'r500', 'image_path', 'expmap_path'.
    mask_catalog_dir : str
        Directory containing mask catalog files ('eRASS1_Main.v1.1.fits' and 'eRASS1_Supp.v1.1.fits').
    output_dir : str
        Directory to save masked image and exposure map files.
    src_radius : float, optional
        Mask radius around sources in arcseconds (default is 100).
    verbose : bool, optional
        If True, print progress and debugging information. Default is True.

    Returns:
    -------
    pd.DataFrame
        New DataFrame with added columns for masked image and exposure map paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load mask catalogs
    mask1 = Table.read(os.path.join(mask_catalog_dir, 'eRASS1_Main.v1.1.fits')).as_array()
    mask2 = Table.read(os.path.join(mask_catalog_dir, 'eRASS1_Supp.v1.1.fits')).as_array()
    mask = np.concatenate((mask1, mask2))
    if mask.shape[0] != mask1.shape[0] + mask2.shape[0]:
        raise ValueError("Combined mask shape does not match expected shape.")

    ra_column, dec_column = mask['RA'], mask['DEC']
    half_size = 4

    # Initialize lists to store paths of masked files
    masked_image_paths = []
    masked_expmap_paths = []

    for index, row in matched_files_df.iterrows():
        source_id = row['ID']
        ra, dec = row['RA'], row['Dec']
        image_path = row['image_path']
        exp_path = row['expmap_path']

        # Generate mask directly in memory
        mask_condition = (
            (ra_column >= ra - half_size) & (ra_column <= ra + half_size) &
            (dec_column >= dec - half_size) & (dec_column <= dec + half_size)
        )
        filtered_mask = mask[mask_condition]

        radius_deg = src_radius / 3600
        distances = np.sqrt((filtered_mask['RA'] - ra)**2 + (filtered_mask['DEC'] - dec)**2)
        final_mask = filtered_mask[distances > radius_deg]

        try:
            # Open image and exposure map files
            with fits.open(image_path) as hdul:
                header = hdul[0].header
                wcs = WCS(header)
                image_data = CCDData(hdul[0].data, unit="adu", header=header)

            with fits.open(exp_path) as hdul:
                exp_data = CCDData(hdul[0].data, unit="adu", header=hdul[0].header)

            ny, nx = image_data.shape
            mask_array = np.ones((ny, nx), dtype=bool)

            for ra_mask, dec_mask, ext_mask in zip(final_mask['RA'], final_mask['DEC'], final_mask['EXT']):
                if ext_mask == 0:
                    mask_radius_deg = 40 / 3600
                elif ext_mask < 60:
                    mask_radius_deg = (ext_mask + 40) / 3600
                else:
                    mask_radius_deg = (ext_mask * 2) / 3600

                region = CircleSkyRegion(
                    center=SkyCoord(ra=ra_mask * u.deg, dec=dec_mask * u.deg, frame='icrs'),
                    radius=mask_radius_deg * u.deg
                )
                region_pix = region.to_pixel(wcs)
                mask_cutout = region_pix.to_mask(mode='center').to_image((ny, nx))
                if mask_cutout is not None:
                    mask_array &= np.logical_not(mask_cutout)

            # Apply mask
            masked_image = image_data.data * mask_array
            masked_exp = exp_data.data * mask_array

            # Save masked results
            output_image_path = os.path.join(output_dir, f"masked_image_id{source_id}.fits")
            output_exp_path = os.path.join(output_dir, f"masked_expmap_id{source_id}.fits")
            fits.writeto(output_image_path, masked_image, image_data.header, overwrite=True)
            fits.writeto(output_exp_path, masked_exp, exp_data.header, overwrite=True)

            if verbose:
                print(f"[{source_id}] Masked image saved: {output_image_path}")
                print(f"[{source_id}] Masked exposure map saved: {output_exp_path}")

            # Append paths to lists
            masked_image_paths.append(output_image_path)
            masked_expmap_paths.append(output_exp_path)

        except Exception as e:
            print(f"[{source_id}] Error processing files: {e}")
            # Append None for failed cases
            masked_image_paths.append(None)
            masked_expmap_paths.append(None)
            continue

    # Create a new DataFrame with additional columns
    updated_matched_files_df = matched_files_df.copy()
    updated_matched_files_df['masked_image_path'] = masked_image_paths
    updated_matched_files_df['masked_expmap_path'] = masked_expmap_paths

    if verbose:
        print("All sources processed and DataFrame updated.")

    return updated_matched_files_df


def calculate_kpc_per_arcsec(z):
    """
    Calculate the kpc per arcsecond at a given redshift.

    Parameters:
    ----------
    z : float
        Redshift of the source.

    Returns:
    -------
    float
        Kpc per arcsecond.
    """
    return cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value



def process_sources_photometry(cleaned_matched_df, verbose=True):
    """
    Process masked images and exposure maps to calculate photometric properties.

    Parameters:
    ----------
    cleaned_matched_df : pd.DataFrame
        DataFrame containing source information and paths to masked image and exposure map files.
    verbose : bool, optional
        If True, print progress. Default is True.

    Returns:
    -------
    pd.DataFrame
        DataFrame with photometric properties for all sources.
    """
    results = []
    arcsec_per_pixel=4.0   # arcsec per pixel

    for i, source in cleaned_matched_df.iterrows():
        source_id = source["ID"]
        source_ra = source["RA"]
        source_dec = source["Dec"]
        source_z = source["z"]

        try:
            # Open masked image and exposure map
            selected_image = source["masked_image_path"]
            selected_expmap = source["masked_expmap_path"]
            img = fits.open(selected_image)
            exp = fits.open(selected_expmap)
            xray_image = img[0].data
            exp_map = exp[0].data
            header = img[0].header
            wcs = WCS(header)

            source_coord = SkyCoord(ra=source_ra * u.degree, dec=source_dec * u.degree, frame='icrs')

            scale_kpc_per_arcsec = calculate_kpc_per_arcsec(source_z)
            scale_kpc_per_pixel = scale_kpc_per_arcsec * arcsec_per_pixel

            r500_pixels = source["r500"] / scale_kpc_per_pixel
            r_bg1 = 3 * r500_pixels
            r_bg2 = 4 * r500_pixels
            r_core_pixels = 0.15 * r500_pixels

            center_pix = wcs.world_to_pixel(source_coord)
            ny, nx = xray_image.shape
            y, x = np.ogrid[:ny, :nx]

            distance_squared = (x - center_pix[0])**2 + (y - center_pix[1])**2

            source_mask_r500 = distance_squared <= r500_pixels**2
            source_mask_r500_excise_core = (distance_squared <= r500_pixels**2) & (distance_squared > r_core_pixels**2)
            background_mask = (distance_squared >= r_bg1**2) & (distance_squared <= r_bg2**2)

            photon_counts_r500 = np.sum(xray_image[source_mask_r500])
            photon_counts_r500_excise_core = np.sum(xray_image[source_mask_r500_excise_core])

            background_counts = np.sum(xray_image[background_mask])
            background_exposure_tot = np.sum(exp_map[background_mask])
            background_counts_avg = background_counts / background_exposure_tot

            exposure_time_r500 = np.mean(exp_map[source_mask_r500])
            exposure_time_r500_excise_core = np.mean(exp_map[source_mask_r500_excise_core])
            exposure_time_r500_tot = np.sum(exp_map[source_mask_r500])
            exposure_time_r500_tot_excise_core = np.sum(exp_map[source_mask_r500_excise_core])

            background_exposure_avg = np.mean(exp_map[background_mask])

            solid_angle_source_r500 = np.sum(source_mask_r500) * (arcsec_per_pixel**2)
            solid_angle_source_r500_excise_core = np.sum(source_mask_r500_excise_core) * (arcsec_per_pixel**2)
            solid_angle_background = np.sum(background_mask) * (arcsec_per_pixel**2)

            solid_angle_ratio_r500 = solid_angle_background / solid_angle_source_r500
            solid_angle_ratio_r500_excise_core = solid_angle_background / solid_angle_source_r500_excise_core

            results.append({
                "ID": source_id,
                "RA": source_ra,
                "Dec": source_dec,
                "z": source_z,
                "M500": source["M500"],
                # "M500_lower": source["M500_lower"],
                # "M500_upper": source["M500_upper"],
                "r500": source["r500"],
                "photon_counts_r500": photon_counts_r500,
                "photon_counts_r500_excise_core": photon_counts_r500_excise_core,
                "background_counts_total": background_counts,
                "background_counts_avg": background_counts_avg,
                "exposure_time_r500_avg": exposure_time_r500,
                "exposure_time_r500_excise_core_avg": exposure_time_r500_excise_core,
                "exposure_time_r500_tot": exposure_time_r500_tot,
                "exposure_time_r500_tot_excise_core": exposure_time_r500_tot_excise_core,
                "background_exposure_avg": background_exposure_avg,
                "solid_angle_source_r500": solid_angle_source_r500,
                "solid_angle_source_r500_excise_core": solid_angle_source_r500_excise_core,
                "solid_angle_ratio_r500": solid_angle_ratio_r500,
                "solid_angle_ratio_r500_excise_core": solid_angle_ratio_r500_excise_core
            })

            if verbose:
                print(f"Processed {i + 1}/{len(cleaned_matched_df)} sources. Example: ID {source_id}")

        except Exception as e:
            print(f"Error processing source ID {source_id}: {e}")

        photon_counts_df = pd.DataFrame(results)

    return photon_counts_df


def calculate_luminosity(photon_counts_df, arf_path, rmf_path, verbose=True):
    """
    Calculate luminosity (Lx) and its error (Lx_err) for each source,
    including Poisson error from both source and background photon counts.

    Parameters:
    ----------
    photon_counts_df : pd.DataFrame
        DataFrame containing source information, including redshift values ('z'),
        photon counts, exposure time, and background information.
    arf_path : str
        Path to the ARF file.
    rmf_path : str
        Path to the RMF file.
    verbose : bool, optional
        If True, print progress and debugging information. Default is True.

    Returns:
    -------
    pd.DataFrame
        Updated DataFrame with added Lx (luminosity) and Lx_err (luminosity error) columns.
    """
    # Check if ARF and RMF files exist
    assert os.path.isfile(arf_path), "ARF file not found"
    assert os.path.isfile(rmf_path), "RMF file not found"

    # Initialize results
    luminosities = []
    luminosity_errors = []

    luminosities_excise_core = []
    luminosity_errors_excise_core = []

    # Cosmology for luminosity distance calculation
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    for index, row in photon_counts_df.iterrows():
        try:
            redshift = row['z']

            # Initialize XSPEC model
            Model('TBabs*apec')
            Xset.abund = 'wilm'
            ms = AllModels(1)
            ms.apec.kT.values = 3.5
            ms.apec.Redshift.values = redshift
            ms.apec.norm.frozen = True
            ms.apec.norm.values = 1
            ms.apec.Abundanc.values = 0.3
            ms.TBabs.nH.values = 0.03
            ms.TBabs.nH.frozen = True

            # Simulate fake spectrum
            AllData.fakeit(
                1,
                FakeitSettings(response=rmf_path, arf=arf_path, exposure=1000, backExposure=0),
                noWrite=True
            )
            AllData.ignore('0-0.2 2.3-**')  # Ignore bands outside 0.2-2.3 keV

            # Calculate count rate
            d1 = AllData(1)
            rate = d1.rate[3]

            # Calculate flux in rest-frame energy band (0.5-2.0 keV)
            emin = 0.5 / (redshift + 1)
            emax = 2.0 / (redshift + 1)
            AllModels.calcFlux(f'{emin} {emax}')

            # Reset absorption to zero and recalculate flux
            ms.TBabs.nH.values = 0.0
            AllModels.calcFlux(f'{emin} {emax}')

            # Calculate ECF
            flux = d1.flux[0]
            ecf = flux / rate

            # Calculate luminosity and error
            photon_r500 = row['photon_counts_r500']
            background_avg = row['background_counts_avg']
            exposure_tot = row['exposure_time_r500_tot']
            exposure_avg = row['exposure_time_r500_avg']
            background_photons = background_avg * exposure_tot

            photon_r500_excise_core = row['photon_counts_r500_excise_core']
            exposure_tot_excise_core = row['exposure_time_r500_tot_excise_core']
            exposure_avg_excise_core = row['exposure_time_r500_excise_core_avg']
            background_photons_excise_core = background_avg * exposure_tot_excise_core

            # Corrected flux
            source_flux = ((photon_r500 - background_avg * exposure_tot) / exposure_avg) * ecf
            # Include background error in total error
            total_photon_error = np.sqrt(photon_r500 + background_photons)
            source_flux_err = (total_photon_error / exposure_avg) * ecf

            # Corrected flux (excise core)
            source_flux_excise_core = ((photon_r500_excise_core - background_avg * exposure_tot_excise_core) / exposure_avg_excise_core) * ecf
            # Include background error in total error
            total_photon_error_excise_core = np.sqrt(photon_r500_excise_core + background_photons_excise_core)
            source_flux_err_excise_core = (total_photon_error_excise_core / exposure_avg_excise_core) * ecf

            lumin_dist = cosmo.luminosity_distance(redshift).value * 3.08568e24  # Convert Mpc to cm
            luminosity = source_flux * 4 * np.pi * lumin_dist**2
            luminosity_err = source_flux_err * 4 * np.pi * lumin_dist**2

            luminosity_excise_core = source_flux_excise_core * 4 * np.pi * lumin_dist**2
            luminosity_err_excise_core = source_flux_err_excise_core * 4 * np.pi * lumin_dist**2

            luminosities.append(luminosity)
            luminosity_errors.append(luminosity_err)

            luminosities_excise_core.append(luminosity_excise_core)
            luminosity_errors_excise_core.append(luminosity_err_excise_core)

            if verbose:
                print(f"[{index + 1}/{len(photon_counts_df)}] Processed source ID: {row['ID']}, Lx: {luminosity}, Lx_err: {luminosity_err}")

        except Exception as e:
            # Handle errors and append None for failed cases
            print(f"Error processing source ID {row['ID']}: {e}")
            luminosities.append(None)
            luminosity_errors.append(None)
            luminosities_excise_core.append(None)
            luminosity_errors_excise_core.append(None)

        finally:
            # Clear XSPEC data and models to avoid interference between sources
            AllData.clear()
            AllModels.clear()

    # Add results to DataFrame
    photon_counts_df['Lx'] = luminosities
    photon_counts_df['Lx_err'] = luminosity_errors
    photon_counts_df['Lx_excise_core'] = luminosities_excise_core
    photon_counts_df['Lx_err_excise_core'] = luminosity_errors_excise_core
    lumin_df = photon_counts_df.copy()

    return lumin_df



# === Main Processing Functions ===

def calculate_luminosity_from_numpy_eRASS(data_array, columns, base_dir, mask_catalog_dir, output_dir, arf_path, rmf_path, src_radius=100, verbose=True):
    """
    Process raw data from a numpy array, apply necessary steps (file matching, masking, etc.),
    and calculate luminosity (Lx) and its error (Lx_err) for each source.

    Parameters:
    ----------
    data_array : np.ndarray
        Raw input data containing source information, at least including 'ID', 'RA', 'Dec', 'z', 'M500'/'M200', 'r500'/'r200'.
    columns : list of str
        Column names corresponding to the data_array, at least including 'ID', 'RA', 'Dec', 'z', 'M500'/'M200', 'r500'/'r200'.
    base_dir : str
        Directory to the ero_archive, e.g., '/path/to/your/ero_archive/'.
    mask_catalog_dir : str
        Directory containing mask catalog files, e.g., '/path/to/your/mask_catalogs/'.
    output_dir : str
        Directory to save intermediate and masked files.
    arf_path : str
        Path to the ARF file.
    rmf_path : str
        Path to the RMF file.
    src_radius : float, optional
        Radius in arcseconds for regions associated with our own sources (default is 100).
    verbose : bool, optional
        If True, print progress and debugging information. Default is True.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing original source information and their Lx, Lx_err.
    """
    # Step 1: Convert numpy array to DataFrame
    photon_counts_df = numpy_to_pandas(data_array, columns)

    # Step 2: Match eROSITA image and exposure map files
    if verbose:
        print("Matching eROSITA files...")
    photon_counts_df = find_eRASS_files(photon_counts_df, base_dir)

    # Step 3: Apply masks to matched files
    if verbose:
        print("Applying masks to sources...")
    photon_counts_df = process_sources_with_mask_eRASS(photon_counts_df, mask_catalog_dir, output_dir, src_radius=100)

    # Step 4: Calculate photometric properties
    if verbose:
        print("Calculating photometric properties for sources...")
    photon_counts_df = process_sources_photometry(photon_counts_df, verbose)

    # Step 5: Calculate luminosity
    if verbose:
        print("Calculating luminosity for sources...")
    lumin_df = calculate_luminosity(photon_counts_df, arf_path, rmf_path, verbose)

    return lumin_df


def calculate_luminosity_from_numpy_eFEDS(data_array, columns, base_dir, mask_catalog_dir, output_dir, arf_path, rmf_path, src_radius=80, verbose=True):
    """
    Process raw data from a numpy array, apply necessary steps (file matching, masking, etc.),
    and calculate luminosity (Lx) and its error (Lx_err) for each source.

    Parameters:
    ----------
    data_array : np.ndarray
        Raw input data containing source information, at least including 'ID', 'RA', 'Dec', 'z', 'M500', 'r500'
    columns : list of str
        Column names corresponding to the data_array, at least including 'ID', 'RA', 'Dec', 'z', 'M500', 'r500'.
    base_dir : str
        Directory to the ero_archive, e.g., '/path/to/your/ero_archive/'.
    mask_catalog_dir : str
        Directory containing mask catalog files, e.g., '/path/to/your/mask_catalogs/'.
    output_dir : str
        Directory to save intermediate and masked files.
    arf_path : str
        Path to the ARF file.
    rmf_path : str
        Path to the RMF file.
    src_radius : float, optional
        Radius in arcseconds for regions associated with our own sources (default is 100).
    verbose : bool, optional
        If True, print progress and debugging information. Default is True.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing original source information and their Lx, Lx_err.
    """
    # Step 1: Convert numpy array to DataFrame
    photon_counts_df = numpy_to_pandas(data_array, columns)

    # Step 2: Match eROSITA image and exposure map files
    if verbose:
        print("Matching eFEDS files...")
    photon_counts_df = find_eFEDS_files(photon_counts_df, base_dir, output_dir, cutout_size_deg=4.0)

    # Step 3: Apply masks to matched files
    if verbose:
        print("Applying masks to sources...")
    photon_counts_df = process_sources_with_mask_eFEDS(photon_counts_df, mask_catalog_dir, output_dir, src_radius=80)

    # Step 4: Calculate photometric properties
    if verbose:
        print("Calculating photometric properties for sources...")
    photon_counts_df = process_sources_photometry(photon_counts_df, verbose)

    # Step 5: Calculate luminosity
    if verbose:
        print("Calculating luminosity for sources...")
    lumin_df = calculate_luminosity(photon_counts_df, arf_path, rmf_path, verbose)

    return lumin_df

