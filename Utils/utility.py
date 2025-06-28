
# Some utility function for searching!

import re

def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_coordinates(text):
    """
    Extract (lat, lon) coordinates from various formats in text:
    - Decimal degrees
    - Decimal + N/S/E/W
    - DMS format
    - 'latitude of ... and longitude of ...'
    - GDA94 / MGA format
    Returns:
        (lat, lon) as float tuple, or None
    """

    # 'latitude of ... and longitude of ...'
    named_decimal = re.search(
        r"latitude\s+(?:is\s+|of\s+)?(-?\d{1,3}(?:\.\d+)?)[^\d]+longitude\s+(?:is\s+|of\s+)?(-?\d{1,3}(?:\.\d+)?)",
        text, re.IGNORECASE)
    if named_decimal:
        return round(float(named_decimal.group(1)), 6), round(float(named_decimal.group(2)), 6)

    # DMS format: 33°52'3.1"S 151°12'25.2"E
    dms_pattern = re.search(
        r"(\d{1,3})[°\s](\d{1,2})['\s](\d{1,2}(?:\.\d+)?)[\"\s]?([NS])\s*(\d{1,3})[°\s](\d{1,2})['\s](\d{1,2}(?:\.\d+)?)[\"\s]?([EW])",
        text, re.IGNORECASE)
    if dms_pattern:
        lat = dms_to_decimal(dms_pattern.group(1), dms_pattern.group(2), dms_pattern.group(3), dms_pattern.group(4).upper())
        lon = dms_to_decimal(dms_pattern.group(5), dms_pattern.group(6), dms_pattern.group(7), dms_pattern.group(8).upper())
        return round(lat, 6), round(lon, 6)

    # Decimal degrees with N/S/E/W: 33.8675° S, 151.2070° E
    dir_decimal = re.search(
        r"(-?\d{1,3}\.\d+)\s*°?\s*([NS])[^,\d]*?(-?\d{1,3}\.\d+)\s*°?\s*([EW])",
        text, re.IGNORECASE)
    if dir_decimal:
        lat = float(dir_decimal.group(1))
        if dir_decimal.group(2).upper() == 'S':
            lat = -lat
        lon = float(dir_decimal.group(3))
        if dir_decimal.group(4).upper() == 'W':
            lon = -lon
        return round(lat, 6), round(lon, 6)

    # Plain decimal degrees: (-33.8675, 151.2070)
    decimal_match = re.search(
        r"\(?\s*(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)\s*\)?", text)
    if decimal_match:
        return round(float(decimal_match.group(1)), 6), round(float(decimal_match.group(2)), 6)

    # GDA94 / MGA style: GDA94/MGA Zone 51, 239944m east, 6410410m north
    gda_match = re.search(r"GDA94.*?(\d{6,7})m east.*?(\d{6,7})m north", text, re.IGNORECASE)
    if gda_match:
        return int(gda_match.group(2)), int(gda_match.group(1))

    return None