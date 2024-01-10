# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image

#logo = 'image/logo.png'
#st.sidebar.image(logo)
# Page title
st.title("Prediction Demo")
# Config


with open('data/step_4_start.pkl', 'rb') as file:
    MIMICtable = pickle.load(file)
    
with open('data/C.pkl', 'rb') as file:
    C = pickle.load(file)
    
with open('data/physpol.pkl', 'rb') as file:
    physpol = pickle.load(file)
    
with open('data/Qon.pkl', 'rb') as file:
    Qon = pickle.load(file)

# Save physician optimal action for each states
phys_OptimalAction = np.argmax(physpol,axis=1).reshape(-1,1)

# Save AI optimal action for each states
OptimalAction=np.argmax(Qon,axis=1).reshape(-1,1)

admission = pd.read_csv("data/admissions.csv")
charts = pd.read_csv("data/charts.csv")
inputs = pd.read_csv("data/inputs.csv")
outputs = pd.read_csv("data/outputs.csv")
labs = pd.read_csv("data/labs.csv")
#==============================================
options = 293325, 294638, 220597, 216859, 232669, 220597, 220597, 217847
user_number = st.selectbox("Select an ICU Stay ID:", options)
inputdata = MIMICtable[MIMICtable['icustayid']== user_number - 200000]
# Reset the index
inputdata = inputdata.reset_index()
inputdata = inputdata.iloc[0]
inputdata['icustayid'] = inputdata['icustayid']+200000
inputdata = inputdata.to_frame().T
inputdata 
# Convert input data to the valid kmeans input
# Mengelompokkan kolom menjadi kategori binary, numerik yang perlu dinormalisasi, dan numerik yang perlu dilog-transform
colbin = ['gender','mechvent','max_dose_vaso','re_admission']
colnorm= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',\
          'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',\
          'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',\
          'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']
collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly']

# Mendapatkan indeks kolom untuk setiap jenis kategori
colbin=np.where(np.isin(inputdata.columns,colbin))[0]
colnorm=np.where(np.isin(inputdata.columns,colnorm))[0]
collog=np.where(np.isin(inputdata.columns,collog))[0]

# Mengambil data yang akan diolah
MIMICraw=inputdata.iloc[:, np.concatenate([colbin,colnorm,collog])].to_numpy()

# Button to trigger prediction
if st.button("Predict"):
    current_state = C.predict(MIMICraw)
        # st.write("current state = ", current_state)
        # Using st.columns to create three columns
    col1, col2, = st.columns(2)

    # Team member 1 in the first column
    with col1:
        physician_action = phys_OptimalAction[current_state]
        st.write("Physician action", physician_action[0])

    # Find AI optimal action
    with col2:
        rec_action = OptimalAction[current_state]
        st.write('Recommended action', rec_action)
        
    vaso_dose = ["0", "0.001–0.08", "0.08–0.22", "0.22–0.45", ">0.45"]
    iv_dose = ["0", "1-50", "50–180", "180–530", ">530"]
    dose_combination = []

    for vaso in vaso_dose:
        for iv in iv_dose:
            dose_combination.append([vaso, iv])

    temp = {
        'IV Fluid': {'Physician': dose_combination[int(physician_action[0])][0], 'AI': dose_combination[int(rec_action)][0]},
        'Vasopressor': {'Physician': dose_combination[int(physician_action[0])][1], 'AI': dose_combination[int(rec_action)][1]}
        
        }
    df = pd.DataFrame(temp)
    st.table(df)
    #=================================================================================================
    st.divider()  
    doses_intensity = ["Zero", "Low", "Medium", "High", "Very High"]
    pair_of_act = list(product(doses_intensity, repeat=2))

    temp = {
        'IV Fluid': {'Physician': pair_of_act[int(physician_action[0])][0], 'AI': pair_of_act[int(rec_action)][0]},
        'Vasopressor': {'Physician': pair_of_act[int(physician_action[0])][1], 'AI': pair_of_act[int(rec_action)][1]}
        }
    df = pd.DataFrame(temp)
    st.table(df)
    img_output = Image.open("image/output.png")
    st.image(img_output, caption="EXample Output", use_column_width=True)
#=====================================================================================================================================
    
# Snip data to specified number of days
    maxdays = 5
    charts = charts.loc[charts.icutime.dt.year<=maxdays]
    outputs = outputs.loc[outputs.icutime.dt.days<=maxdays]
    inputs = inputs.loc[inputs.icustarttime.dt.days<=maxdays]
    labs = labs.loc[labs.icutime.dt.days<=maxdays]

    # Create column with minutes from ICU intime
    charts['icutimehr'] = (charts['icutime'].dt.seconds/60/60)+(charts['icutime'].dt.days*24)
    outputs['icutimehr'] = (outputs['icutime'].dt.seconds/60/60)+(outputs['icutime'].dt.days*24)
    inputs['icustarttimehr'] = (inputs['icustarttime'].dt.seconds/60/60)+(inputs['icustarttime'].dt.days*24)
    inputs['icuendtimehr'] = (inputs['icuendtime'].dt.seconds/60/60)+(inputs['icuendtime'].dt.days*24)
    labs['icutimehr'] = (labs['icutime'].dt.seconds/60/60)+(labs['icutime'].dt.days*24)

    # Get average values
    hr_mean = charts.valuenum[charts.label=='Heart Rate'].mean()
    bp_mean = charts.icutimehr[charts.label=='Non Invasive Blood Pressure mean'].mean()
    temp_mean = ((charts.valuenum[charts.label=='Temperature Fahrenheit']-32)/1.8).mean()

    # Plot sample data over first 24 hours from admission to ICU
    # Credit: Randal Olson for styling (http://www.randalolson.com/2014/06/28/)

    # Prepare the size of the figure
    fig = plt.figure(figsize=(22, 20))
    plt.rcParams.update({'font.size': 22})

    # "Tableau 20" colors as RGB.   
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
    
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)
        
    # Remove the plot frame lines. 
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(True)    
    
    # Ensure that the axis ticks only show up on the bottom and left of the plot.      
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    ax.axis([0,60,0,225])

    # Plot vital signs
    plt.plot(charts.icutimehr[charts.label=='Heart Rate'], 
            charts.valuenum[charts.label=='Heart Rate'],
            color=tableau20[6], lw=2.5,
            marker='o', markersize=6, label='Heart rate')

    plt.plot(charts.icutimehr[charts.label=='O2 saturation pulseoxymetry'], 
            charts.valuenum[charts.label=='O2 saturation pulseoxymetry'],
            color=tableau20[1], lw=2.5, 
            marker='o', markersize=6, label='O2 saturation')

    plt.plot(charts.icutimehr[charts.label=='Non Invasive Blood Pressure mean'], 
            charts.valuenum[charts.label=='Non Invasive Blood Pressure mean'],
            color=tableau20[4], lw=2.5,
            marker='o', markersize=6, label='NIBP, mean')

    plt.plot(charts.icutimehr[charts.label=='Respiratory Rate'], 
            charts.valuenum[charts.label=='Respiratory Rate'],
            color=tableau20[2], lw=2.5,
            marker='o', markersize=6, label='Respiratory rate')

    # for i, txt in enumerate(charts.value[charts.label=='Temperature Fahrenheit'].values):
    #         plt.annotate(txt,(charts.icutimehr[charts.label=='Temperature Fahrenheit'].
    #                            values[i],140),fontsize=15)

    # Plot input/output events
    plt.plot(inputs.icustarttimehr[inputs.amountuom=='mL'], 
            inputs.amount[inputs.amountuom=='mL'].cumsum()/100, 
            color=tableau20[9], lw=2.5,
            marker='o', markersize=6, label='Intake volume, dL')

    plt.plot(outputs.icutimehr, 
            outputs.value.cumsum()/100, 
            color=tableau20[10], lw=2.5,
            marker='o', markersize=6, label='Output volume, dL')

    # Plot intravenous meds
    plt.text(-10,150,'NaCl 0.9%',fontsize=17)
    for i,row in inputs.loc[(inputs["label"] =='NaCl 0.9%') & (inputs["rate"] > 0)].iterrows():
        plt.plot([row['icustarttimehr'],row['icuendtimehr']],[150]*2,
                color=tableau20[16], lw=4,marker='o', markersize=6)
        plt.text(row['icustarttimehr'],150,
                str(round(row['rate'],1)) + ' ' + str(row['rateuom']),
                fontsize=15)

    plt.text(-10,145,'Amiodarone',fontsize=17)
    for i,row in inputs.loc[(inputs["label"] =='Amiodarone 600/500') & (inputs["rate"] > 0)].iterrows():
        plt.plot([row['icustarttimehr'],row['icuendtimehr']],[145]*2,
                color=tableau20[16], lw=4,marker='o', markersize=6)
        plt.text(row['icustarttimehr'],145,
                str(round(row['rate'],1)) + ' ' + str(row['rateuom']),
                fontsize=15)    

    plt.text(-10,140,'Dextrose 5%',fontsize=17)
    for i,row in inputs.loc[(inputs["label"] =='Dextrose 5%') 
                            & (inputs["rate"] > 0) & (inputs["rate"] < 500)].iterrows():
        plt.plot([row['icustarttimehr'],row['icuendtimehr']],[140]*2,
                color=tableau20[16], lw=4,marker='o', markersize=6)
        plt.text(row['icustarttimehr'],140,
                str(round(row['rate'],1)) + ' ' + str(row['rateuom']),
                fontsize=15)    

    plt.text(-10,165,'Morphine Sulfate',fontsize=17)
    plt.plot(inputs.icustarttimehr[inputs.label=='Morphine Sulfate'],
            [165]*len(inputs[inputs.label=='Morphine Sulfate']),
            color=tableau20[16], lw=0, marker='o', markersize=6)   
        
    plt.text(-10,160,'Vancomycin (1 dose)',fontsize=17)
    plt.plot(inputs.icustarttimehr[inputs.label=='Vancomycin'],
            [160]*len(inputs[inputs.label=='Vancomycin']),
            color=tableau20[16], lw=0, marker='o', markersize=6)
        
    plt.text(-10,155,'Piperacillin (1 dose)',fontsize=17)
    plt.plot(inputs.icustarttimehr[inputs.label=='Piperacillin/Tazobactam (Zosyn)'],
            [155]*len(inputs[inputs.label=='Piperacillin/Tazobactam (Zosyn)']),
            color=tableau20[16], lw=0, marker='o', markersize=6)


    # Plot labs
    plt.text(-10,175,'Neutrophil, %',fontsize=17)
    for i, txt in enumerate(labs.value[labs.label=='NEUTROPHILS'].values):
            plt.annotate(txt, (labs.icutimehr[labs.label=='NEUTROPHILS'].
                            values[i],175),fontsize=17) 

    plt.text(-10,180,'White blood cell, K/uL',fontsize=17)
    for i, txt in enumerate(labs.value[labs.label=='WHITE BLOOD CELLS'].values):
            plt.annotate(txt, (labs.icutimehr[labs.label=='WHITE BLOOD CELLS'].
                            values[i],180),fontsize=17)

    plt.text(-10,185,'Creatinine, mg/dL',fontsize=17)        
    for i, txt in enumerate(labs.value[labs.label=='CREATININE'].values):
            plt.annotate(txt, (labs.icutimehr[labs.label=='CREATININE'].
                            values[i],185),fontsize=17)

    plt.text(-10,190,'Platelet, K/uL',fontsize=17)
    for i, txt in enumerate(labs.value[labs.label=='PLATELET COUNT'].values):
            plt.annotate(txt, (labs.icutimehr[labs.label=='PLATELET COUNT'].
                            values[i],190),fontsize=17)

    # Plot Glasgow Coma Scale
    plt.text(-10,200,'GCS: Eye',fontsize=17)
    for i, txt in enumerate(charts.value[charts.label=='GCS - Eye Opening'].values):
        if np.mod(i,2)==0 and i < 65:
            plt.annotate(txt, (charts.icutimehr[charts.label=='GCS - Eye Opening'].
                            values[i],200),fontsize=17)

    plt.text(-10,205,'GCS: Motor',fontsize=17)
    for i, txt in enumerate(charts.value[charts.label=='GCS - Motor Response'].values):
        if np.mod(i,2)==0 and i < 65:
            plt.annotate(txt, (charts.icutimehr[charts.label=='GCS - Motor Response'].
                            values[i],205),fontsize=17)

    plt.text(-10,210,'GCS: Verbal',fontsize=17)  
    for i, txt in enumerate(charts.value[charts.label=='GCS - Verbal Response'].values):
        if np.mod(i,2)==0 and i < 65:
            plt.annotate(txt, (charts.icutimehr[charts.label=='GCS - Verbal Response'].
                            values[i],210),fontsize=17)

    # Plot code status
    plt.text(-10,220,'Code status',fontsize=17) 
    for i, txt in enumerate(charts.value[charts.label=='Code Status'].values):
            plt.annotate(txt, (charts.icutimehr[charts.label=='Code Status'].
                            values[i],220),fontsize=17)
            
    plt.legend(loc=5,fontsize=18)
    plt.xlabel('Time after admission to the intensive care unit, hours', fontsize=22)
    plt.ylabel('Measurement, absolute value', fontsize=22)
    plt.yticks(np.arange(0, 140, 20))

    # Save the figure
    fig.savefig('examplepatient.pdf', bbox_inches='tight')
    #======================================================
   
#================================================================================  
st.divider() 
st.write("""
Copyright © 2023 - SOLID, All Rights Reserved.
""")

