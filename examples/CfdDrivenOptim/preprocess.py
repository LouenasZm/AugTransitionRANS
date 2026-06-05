"""
    Author: Louenas Zemmour
    Date:   August 2025

    Module to preprocess ERCOFTAC dataset, used before training surrogate models 
    and to interpolate LES and RANS in the same mesh to compute loss. 

    The computation of the loss is done only for the boundary layer as the freestream is
    assumed to be the same, the boundary layer is determined by the value of the the normalized
    strain rate being less than 0.5
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate  import griddata
from ppModule.interface          import PostProcessMusicaa

# Define bounds for Ercoftac dataset
ERCOFTAC_BOUNDS = {
    "T3A" : {
        "x": (0.0, 0.035),
        "y": (0.0, 0.001)
    },

    "T3B": {
        "x": (0.0, 0.030),
        "y": (0.0, 0.001)
    },

    "T3C2": {
        "x": (0.0, 0.126),
        "y": (0.0, 0.007)
    },

    "T3C3": {
        "x": (0.0, 0.096),
        "y": (0.0, 0.006)
    },

    "T3C4": {
        "x": (0.0, 0.033),
        "y": (0.0, 0.002)
    },

    "T3C5": {
        "x": (0.0, 0.21),
        "y": (0.0, 0.008)
    }
}

def fill_nans_with_interpolation(data_dict, x_grid, y_grid):
    """Interpolate NaN values in the data dictionary using nearest neighbor interpolation."""
    filled_data = {}
    for key, arr in data_dict.items():
        arr = np.array(arr)
        if np.isnan(arr).any():
            # Get coordinates of valid and nan points
            x_flat = x_grid.flatten()
            y_flat = y_grid.flatten()
            arr_flat = arr.flatten()
            valid = ~np.isnan(arr_flat)
            nan = np.isnan(arr_flat)
            # Interpolate only at nan points using nearest neighbor
            arr_flat[nan] = griddata(
                points=np.column_stack((x_flat[valid], y_flat[valid])),
                values=arr_flat[valid],
                xi=np.column_stack((x_flat[nan], y_flat[nan])),
                method='nearest'
            )
            arr = arr_flat.reshape(arr.shape)
        filled_data[key] = arr
    return filled_data

class ErcoftacPreprocess:
    """
    Class to preprocess ERCOFTAC dataset.
    
    This class contains methods to preprocess LES and RANS data for different cases.
    It uses predefined bounds for each case to mask and interpolate the data.
    """

    def __init__(self, case: str, config: dict = None):
        """
        Initialize the ErcoftacPreprocess class with a case identifier.
        Args:
            case (str): The case identifier (e.g., "T3A", "T3B", etc.).
        """
        self.config = config
        self.case = case
        self._set_bounds()
        self._set_baseline()

    # ================ Public Methods ==========================
    def preprocess_les(self, grid: dict, stats: dict, keylist: list) -> dict:
        """
        Preprocess the LES data for the initialized case.
        
        Args:
            grid (dict): Grid dictionary containing LES data.
            stats (dict): Statistics dictionary containing LES data.
        
        Returns:
            dict: Processed LES data.
        """
        #
        keylist.append("magn_S")
        keylist.append("vorticity")
        #
        # Mask data out of bounds:
        # Mask only keys listed in keylist (leave others unchanged)
        keylist_ = [k for k in keylist if k != "cf"]
        for k in keylist:
            if k in stats[2]:
                vals = np.array(stats[2][k])
                # mask points outside y and x bounds, then fill masked entries with NaN
                vals = np.ma.masked_where(grid["y"][2] > self.bounds["y"][1], vals)
                vals = np.ma.masked_where(grid["x"][2] > self.bounds["x"][1], vals)
                stats[2][k] = vals.filled(np.nan)

        s11 = stats[2]['rho*dux']/ stats[2]['rho']
        s22 = stats[2]['rho*duy']/ stats[2]['rho']
        s12 = 0.5 * (stats[2]['rho*dvx'] + stats[2]['rho*duy']) / stats[2]['rho']
        # Plot velocity before interpolation:
        stats[2]["magn_S"]      = np.sqrt(s11**2 + s22**2 + 2 * s12**2)
        stats[2]["vorticity"]   = (stats[2]["rho*duy"] - 
                                       stats[2]["rho*dvx"]) / stats[2]["rho"]
        # Interpolate to a unifom grid:
        points = np.column_stack((grid["x"][2].flatten(), grid["y"][2].flatten()))
        processed_data = {}
        for key in keylist:
            if key in stats[2]:
                vals = np.array(stats[2][key]).flatten()
                processed_data[key] = griddata(points, vals, (self.x, self.y), method='linear')
            else:
                processed_data[key] = np.full_like(self.x, np.nan, dtype=float)
        processed_data["S_star"] = processed_data["magn_S"] / (processed_data["magn_S"]
                                                               + processed_data["vorticity"])
        # processed_data = fill_nans_with_interpolation(processed_data, self.x, self.y)
        #
        # Interpolate Cf:
        processed_data["cf"] = np.interp(self.x[:,0], grid["x"][2][:,0], stats[2]["cf"])
        # Interpolate Ue:
        if "ufst" in stats[2]:
            processed_data["ufst"] = np.interp(self.x[:,0], grid["x"][2][:,0], stats[2]["ufst"])
        # Plot LES
        return processed_data
    #
    #
    #
    def preprocess_rans(self, grid: dict, stats: dict, keylist: list) -> dict:
        """
        Preprocess the RANS data for the initialized case.
        
        Args:
            grid (dict): Grid dictionary containing RANS data.
            stats (dict): Statistics dictionary containing RANS data.
        
        Returns:
            dict: Processed RANS data.
        """
        # Get grid:
        x, y = grid["x"][1], grid["y"][1]
        #
        # Mask data out of bounds:
        keylist_ = [k for k in keylist if k != "cf"]
        for k in keylist_:
            if k in stats[1]:
                vals = np.array(stats[1][k])
                # mask points outside y and x bounds, then fill masked entries with NaN
                vals = np.ma.masked_where(y > self.bounds["y"][1], vals)
                vals = np.ma.masked_where(x > self.bounds["x"][1], vals)
                stats[1][k] = vals.filled(np.nan)

        processed_data = {}
        processed_data["cf"] = stats[1]["cf"]
        for k in keylist:
            processed_data[k] = stats[1][k]
        return processed_data

    # ================ Private Methods =========================
    def _set_bounds(self):
        """
        Set the bounds for the ERCOFTAC dataset based on the case identifier.
        
        Raises:
            ValueError: If the case identifier is not defined in ERCOFTAC_BOUNDS.
        """
        if self.case.upper() not in ERCOFTAC_BOUNDS:
            raise ValueError(f"Case {self.case} is not defined in ERCOFTAC_BOUNDS.")
        self.bounds = ERCOFTAC_BOUNDS[self.case.upper()]

    def _set_baseline(self):
        """
        Read baseline RANS simulation to interpolate LES data into RANS mesh.
        """
        config = deepcopy(self.config)
        pp_baseline = PostProcessMusicaa(config[self.case])
        x_baseline = pp_baseline.config["grid"]["x"][1]
        y_baseline = pp_baseline.config["grid"]["y"][1]
        # Mask baseline grid out of bounds
        self.x = np.ma.masked_where(y_baseline > self.bounds["y"][1], x_baseline)
        self.y = np.ma.masked_where(y_baseline > self.bounds["y"][1], y_baseline)
        self.x = np.ma.masked_where(x_baseline > self.bounds["x"][1], self.x)
        self.y = np.ma.masked_where(x_baseline > self.bounds["x"][1], self.y)
