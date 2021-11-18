# A quick python script for downloading the appropriate data files from FigShare
# This stores the files in /data, which is ignored in .gitignore
import os
import requests

# Get file path relative to this script
repo_path = os.path.dirname(os.path.realpath(__file__))
print('Detected repository path: ', repo_path)

data_path = os.path.join(repo_path, 'data')

# Make /data if it does not exist
print('Creating a data folder: ', data_path)
try:
    os.mkdir(data_path)
except Exception as e:
    print('Failed creating data folder. See error below:')
    print(e)

# Download all the files into /data
print('Getting data files from FigShare...')

files = [
    {'url': 'https://figshare.com/ndownloader/files/31521599', 'name': 'lenvar_full_500samples_trimmed.pkl'},
    {'url': 'https://figshare.com/ndownloader/files/31511888', 'name': 'tsvar_all495hours_10000samples_trimmed.pkl'},
    {'url': 'https://figshare.com/ndownloader/files/31511891', 'name': 'tsvar_load495hours_1000samples_trimmed.pkl'},
    {'url': 'https://figshare.com/ndownloader/files/31511885', 'name': 'tsvar_solar495hours_1000samples_trimmed.pkl'},
    {'url': 'https://figshare.com/ndownloader/files/31511897', 'name': 'tsvar_wind495hours_1000samples_trimmed.pkl'},
    {'url': 'https://figshare.com/ndownloader/files/31511894', 'name': 'LHS_20000samples_495hours_trimmed.pkl'}
]

for file in files:
    print('\tDownloading ', file['name'], '...')
    try:
        file_data = requests.get(file['url'], allow_redirects=True)
        with open(os.path.join(data_path, file['name']), 'wb') as f:
            f.write(file_data.content)
        print('\t\tdone')
    except Exception as e:
        print('\t\tfailed')
        print(e)
    

