import polars as pl
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

def fit_spline(series, years, parse=False):
        """
        Applies PHCHIP interpolation to avoid extreme oscillations.
        """
        df = pl.DataFrame([years,series]).filter(pl.exclude('year').is_not_null())

        # Beginning year is the first year with non-null value
        granu_years = np.arange(df['year'].item(0), years.max() + 1)

        # If parse is True, parse df's years are divisible by 10
        if parse is True:
            df = df.filter(pl.col('year') % 10 == 0)

        fit_years = df['year'].to_numpy()
        fit_values = df.select(pl.exclude('year')).to_numpy()

        spline = PchipInterpolator(fit_years, fit_values)
        # Predict for all years using the fitted spline interpolation
        granu_values = spline(granu_years)

        # Ensure all values are non-negative
        granu_values = np.maximum(granu_values, 0)
        granu_values = granu_values.ravel()  # Convert to 1D array

        # pad with 0s if the first year is not the beginning year
        granu_values = np.pad(granu_values, (granu_years[0] - years[0], 0), 'constant', constant_values=0)

        return granu_values

def fit_linear(series, years, parse=False):
        """
        Applies linear interpolation instead of spline interpolation.
        """
        df = pl.DataFrame([years,series]).filter(pl.exclude('year').is_not_null())

        # Beginning year is the first year with non-null value
        granu_years = np.arange(df['year'].item(0), years.max() + 1)

        # If parse is True, parse df's years are divisible by 10
        if parse is True:
            df = df.filter(pl.col('year') % 10 == 0)

        fit_years = df['year'].to_numpy()
        fit_values = df.select(pl.exclude('year')).to_numpy().ravel()

        # Use linear interpolation
        if len(fit_years) > 1:
            interp_func = interp1d(fit_years, fit_values, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            granu_values = interp_func(granu_years)
        else:
            # If only one data point, use constant value
            granu_values = np.full(len(granu_years), fit_values[0])

        # Ensure all values are non-negative
        granu_values = np.maximum(granu_values, 0)

        # pad with 0s if the first year is not the beginning year
        granu_values = np.pad(granu_values, (granu_years[0] - years[0], 0), 'constant', constant_values=0)

        return granu_values