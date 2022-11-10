import requests

url = "http://0.0.0.0:9696/predict"
appliance = {
    "t1": 18.963333333333296,
    "t2": 18.29,
    "t3": 19.89,
    "t4": 18.5,
    "t5": 17.79,
    "t7": 18.1,
    "t8": 19.29,
    "rh_1": 42.09,
    "rh_2": 41.29,
    "rh_3": 43.2,
    "rh_4": 41.79,
    "rh_5": 64.6233333333333,
    "rh_6": 93.1266666666667,
    "rh_7": 42.09,
    "rh_8": 51.09,
    "rh_9": 46.03,
    "t_out": 2.86666666666667,
    "tdewpoint": 1.68333333333333,
    "rh_out": 92.3333333333333,
    "press_mm_hg": 753.733333333333,
    "windspeed": 5.16666666666667,
    "visibility": 30.0,
    "date_dow": 3.0,
    "date_dom": 14.0,
    "date_doy": 14.0,
    "date_hr": 2.0,
    "date_min": 10.0,
    "date_wkoyr": 2.0,
    "date_mth": 1.0,
    "date_qtr": 1.0
}
print(requests.post(url, json=appliance).json())