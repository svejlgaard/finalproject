import matplotlib.pyplot as plt
import numpy as np
import os, contextlib, sys
from loaddata import loaddata
import datetime
import math

# Sets the directory to the current directory
os.chdir(sys.path[0])

# From the Navy's Astronomical Equation:
def get_julian_datetime(date):
    """
    Convert a datetime object into julian float.
    Args:
        date: datetime-object of date in question

    Returns: float - Julian calculated datetime.
    Raises: 
        TypeError : Incorrect parameter type
        ValueError: Date out of range of equation
    """

    # Ensure correct format
    if not isinstance(date, datetime.datetime):
        raise TypeError('Invalid type for parameter "date" - expecting datetime')
    elif date.year < 1801 or date.year > 2099:
        raise ValueError('Datetime must be between year 1801 and 2099')

    # Perform the calculation
    julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int(
        (275 * date.month) / 9.0) + date.day + 1721013.5 + (
                          date.hour + date.minute / 60.0 + date.second / math.pow(60,
                                                                                  2)) / 24.0 - 0.5 * math.copysign(
        1, 100 * date.year + date.month - 190002.5) + 0.5

    return julian_datetime

directory = 'extended_data'

for filename in os.listdir(directory):
    data = np.genfromtxt(f'{directory}/{filename}',delimiter=',',skip_header=1,dtype=str)
    final_data = np.genfromtxt(f'{directory}/{filename}',delimiter=',',skip_header=1,dtype=float)
    if 'build' in filename:
        time_list = list(data[:,0].copy())
        time_list = [datetime.datetime(int(i.strip('"').split("-")[0]),
                    int(i.split("-")[1]),
                    int(i.split("-")[2].split(" ")[0]),
                    int(i.split("-")[2].split(" ")[1].split(":")[0]),
                    int(i.split("-")[2].split(" ")[1].split(":")[1]),
                    int(i.strip('"').split("-")[2].split(" ")[1].split(":")[2])) for i in time_list]
        final_data[:,0] = [get_julian_datetime(i) for i in time_list]

