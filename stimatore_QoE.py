import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import tree
import time
import pyautogui
from datetime import datetime


#addestramento del modello di machine learning

dataset = pd.read_pickle("oversampled_dataset")
x_train, x_test, y_train, y_test = train_test_split(dataset,dataset['QoE'],test_size=0.35)

RFC = RandomForestClassifier()
RFC = RFC.fit(x_train,y_train)
prediction = RFC.predict(x_test)

print("Random Forest model trained with accuracy:",accuracy_score(y_test, prediction))
print("\nStarting reading extracted features and determining QoE...")

needed_features = ['frame','gaze_0_x','gaze_0_y','gaze_1_x','gaze_1_y','gaze_angle_x','gaze_angle_y','AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c','AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c','AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c','AU01_r', 'AU02_r', 'AU04_r', 'AU07_r','AU09_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r','AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']

total_AUcs = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c','AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c','AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

needed_gaze_feat = ['gaze_0_x','gaze_0_y','gaze_1_x','gaze_1_y','gaze_angle_x','gaze_angle_y']

needed_AUr_feat = ['AU01_r', 'AU02_r', 'AU04_r', 'AU07_r','AU09_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r','AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']

needed_AUc_feat = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU09_c', 'AU10_c', 'AU15_c', 'AU17_c', 'AU23_c', 'AU28_c', 'AU45_c']


processed_frames = 0
computed_values_sumup = pd.DataFrame()

while(True):
    try:
        #lettura delle feature estratte dal volto dell'utente inquadrato della telecamera
        features = pd.read_csv("/home/gabriele/Scrivania/Tirocinio/prog/prova_modello/OpenFace_features_extracted/extracted_features.csv",usecols=needed_features,skiprows=[i for i in range(1,processed_frames+1)])
        features = features.dropna()
        if features.empty==True:
            raise pd.errors.EmptyDataError()
        gaze = features[needed_gaze_feat]
        AUc = features[needed_AUc_feat]
        AUr = features[needed_AUr_feat]
        
    except FileNotFoundError:
        print("Live feature extraction not started yet! Features file not found...")
        time.sleep(5)
    except pd.errors.EmptyDataError:
        print('\n')
        print("Live feature extraction ended! Features file is empty...")
        print('\n')
        computed_values_sumup.to_csv("computed_QoE__"+str(datetime.now()))
        pyautogui.alert("Live feature extraction ended. Feature file is empty.\n\nProducing results file...\n\nTerminating QoE estimation program...",title="QoE estimation ended",timeout=6000)
        break
    except ValueError:
        print("Error! Value not recognised in features file.")
        break

    else:        #le eccezioni non si verificano; l'estrazione delle feature dal volto è iniziata correttamente
        
        #calcolo delle metriche da utilizzare per determinare la QoE dell'utente
        gaze_directions_sum = gaze.sum().sum()
        gaze_directions_number = gaze.shape[0]*gaze.shape[1]
        gaze_directions_mean = gaze_directions_sum/gaze_directions_number
        gaze = gaze.subtract(gaze_directions_mean)
        final_sum = gaze.sum().sum()
        FEATURE_gaze_directions_variance = final_sum/gaze.shape[0]
    
        sum_dic = {}
        total_auc_sum = 0
        for auc in total_AUcs:
            temp = features[auc]
            auc_sum = temp.sum()
            sum_dic[auc] = auc_sum
            total_auc_sum = total_auc_sum + auc_sum
        freq_dic = {}
        for AUC in needed_AUc_feat:
            freq_dic[AUC] = sum_dic[AUC]/total_auc_sum
        FEATURE_AUc_frequency = pd.DataFrame()
        FEATURE_AUc_frequency = FEATURE_AUc_frequency.append(freq_dic,ignore_index=True)
    
        sum_aur_dic = {}
        for aur in needed_AUr_feat:
            temp = features[aur]
            aur_sum = temp.sum()
            sum_aur_dic[aur] = aur_sum
        FEATURE_AUr_intensity = pd.DataFrame()
        FEATURE_AUr_intensity = FEATURE_AUr_intensity.append(sum_aur_dic,ignore_index=True)
        
        MODEL_INPUT_FEATURES = pd.DataFrame()
        d = {}
        d['Var'] = FEATURE_gaze_directions_variance
        var = pd.DataFrame()
        var = var.append(d,ignore_index=True)
        q = {}
        q['QoE'] = -1
        qual = pd.DataFrame()
        qual = qual.append(q,ignore_index=True)
        MODEL_INPUT_FEATURES = pd.concat([var,FEATURE_AUc_frequency,FEATURE_AUr_intensity,qual],axis=1)
    
        #determinazione della quality of experience dell'utente
        
        computed = {}
        computed_QoE = pd.DataFrame()
        
        pred = RFC.predict(MODEL_INPUT_FEATURES)
        
        computed['Video frames'] = str(processed_frames+1)+' : '+str(processed_frames + features.shape[0])
        computed['QoE prediction'] = pred[0]
        computed_QoE = computed_QoE.append(computed,ignore_index=True)
        
        aux = pd.concat([computed_QoE,var,FEATURE_AUc_frequency,FEATURE_AUr_intensity],axis=1)
        computed_values_sumup = computed_values_sumup.append(aux)
        
        processed_frames = processed_frames + features.shape[0]
        
        print('\n')
        print('Statistiche: ',computed_QoE)
        
        pyautogui.alert("Your estimated Quality of Experience is now:\n\n"+str(pred[0]),title="QoE estimation output",timeout=3500)
        
        time.sleep(10)



