import json
import sys
sys.path.append('../')

import re

def basic_time_normalizer(text):
    check_year, check_month, check_day = False, False, False
    # find year
    regex = re.compile("(?P<YEAR>\d+)년")
    y_search = regex.search(text)
    if y_search:
        year = y_search.group('YEAR')
        check_year = True
    else:
        year = '0000'
    
    # find month
    regex = re.compile("(?P<MONTH>\d+)월")
    m_search = regex.search(text)
    if m_search:
        month = m_search.group('MONTH')
        check_month = True
    else:
        month = '00'
    
    # find day
    regex = re.compile("(?P<DAY>\d+)일")
    d_search = regex.search(text)
    if d_search:
        day = d_search.group('DAY')
        check_day = True
    else:
        day = '00'
    
    if check_year or check_month or check_day:
        if len(year) == 4 and len(month) <= 2 and len(day) <=2:
            time_rep = year+'-'+month+'-'+day
        else:
            time_rep = text
    else:
        time_rep = text    
    return time_rep    

def time2xsd(text):
    time_rep = basic_time_normalizer(text)
    
    return time_rep