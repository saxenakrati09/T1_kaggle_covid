import sys
sys.path.append('../')
import os
os.chdir("..")
print("cwd ", os.getcwd())
cwd =  os.getcwd()
from Init.PropertiesReader import PropertyFileReader
#from dm.init.PropertiesReader import PropertyFileReader

'''
Set the path to the metaMTConfig file.
'''
#windows
metaMTConfig = os.path.join(cwd, 'MetaMTConfig.properties')
'''
It is expected that both the meta* and the actual config files are in the same folder and inside the project folder.
If this assumption is invalid, then change the code below to accommodate.
'''
def getConfigFilePath(metaMTConfig):
    print(metaMTConfig)
    p = PropertyFileReader()
    p.load(metaMTConfig)
    configFileName = p['currentConfigFile']
    #configFileName = p['project'] #print metaMTConfig.split("/")
    #configPath = metaMTConfig.replace(metaMTConfig.split("/")[-1], configFileName)
    #print "current config path: ", configPath
    #return configPath
    print("current config path: ", configFileName)
    return configFileName

def initAllPaths(metaMTConfig):
    configPath = getConfigFilePath(metaMTConfig)
    p = PropertyFileReader()
    p.load(configPath)
    home = p["home"] #store home with ending /

    if home=='':
        home = cwd
        print(home)

    biorxiv_dir = home + p['biorxiv_dir']
    pmc_dir = home + p['pmc_dir']
    comm_dir = home + p['comm_dir']
    noncomm_dir = home + p['noncomm_dir']
    biorxiv_csv_path = home + p['biorxiv_csv_path']
    pmc_csv_path = home + p['pmc_csv_path']
    comm_csv_path = home + p['comm_csv_path']
    noncomm_csv_path = home + p['noncomm_csv_path']

    biorxiv_cleaner_csv_path = home + p['biorxiv_cleaner_csv_path']
    pmc_cleaner_csv_path = home + p['pmc_cleaner_csv_path']
    comm_cleaner_csv_path = home + p['comm_cleaner_csv_path']
    noncomm_cleaner_csv_path = home + p['noncomm_cleaner_csv_path']
    all_papers_cleaner_path = home + p['all_papers_cleaner_path']

    root_path = home + p['root_path']
    metadata_path = home + p['metadata_path']
    cord19_df_csv_path = home + p['cord19_df_csv_path']

    bert_mean_token_path = home + p['bert_mean_token_path']

    return biorxiv_dir, pmc_dir, comm_dir, noncomm_dir, biorxiv_csv_path, pmc_csv_path, comm_csv_path, noncomm_csv_path, root_path, metadata_path, cord19_df_csv_path, bert_mean_token_path, biorxiv_cleaner_csv_path, pmc_cleaner_csv_path, comm_cleaner_csv_path, noncomm_cleaner_csv_path, all_papers_cleaner_path


biorxiv_dir, pmc_dir, comm_dir, noncomm_dir, biorxiv_csv_path, pmc_csv_path, comm_csv_path, noncomm_csv_path, root_path, metadata_path, cord19_df_csv_path, bert_mean_token_path, biorxiv_cleaner_csv_path, pmc_cleaner_csv_path, comm_cleaner_csv_path, noncomm_cleaner_csv_path, all_papers_cleaner_path=initAllPaths(metaMTConfig)

