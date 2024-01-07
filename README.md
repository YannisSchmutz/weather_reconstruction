# weather_reconstruction


## Local Execution
```
streamlit run app.py
```
http://localhost:8501

## Deployment

### Create VENV

```
python3.11 -m venv venv_wr
```

### Access VENV
```
source venv_wr/bin/activate
```

### Install needed packages
```
pip install streamlit
pip install matplotlib
pip install numpy
pip install xarray
pip install pandas

pip install pyyaml
pip install netcdf4
pip install h5netcdf
pip install cartopy
pip install statsmodels

pip install pipreqs
```

### Create requirements.txt
```
pipreqs .
```