# Early Prediction of Sepsis From Clinical Data

1. [Résumé](#Résumé)
2. [Objectives](#Objectives)
3. [The Data](#The-Data)

- Try this Binder for **EDA** : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nevermind78/capstone/main?filepath=EDA.ipynb)

- Try this Binder for **Presentation** : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nevermind78/capstone/main?filepath=presentation.ipynb)

- Try this Binder for **Demonstartion** : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nevermind78/capstone/main?filepath=Demo.ipynb)

- **Six ML Algorithm** [![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nevermind78/Capstone/blob/main/sepsis_lr.ipynb)

- **MLP**  [![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nevermind78/Capstone/blob/main/MLP2.ipynb)


## Résumé : 
Intensive care units (ICU) are the part of hospitals where decision making is very important, complex and critical at the same time. Deciding correctly and at the right time who is the patient who needs critical care and who does not can reduce the pressure on these units, ensuring good management and allocation of resources which are both limited and very expensive. This good management will guarantee a considerable saving in costs by reducing them considerably. In fact, people whose condition is critically ill as a result of accidents, surgical complications, serious respiratory problems, infections and serious injuries can achieve death if they are not treated with technology and sophisticated equipment allowing close and constant monitoring and maintenance of bodily functions. In particular, sepsis is a major health problem and life-threatening illness that occurs when the body's response to infection causes tissue damage, organ failure, or death. Worldwide, around 30 million people develop sepsis and 6 million people die from sepsis each year. The costs of managing sepsis in hospitals exceed those of any other health problem. The majority of these costs relate to patients who develop sepsis while in hospital. Early detection and antibiotic treatment of sepsis improves outcomes. However, although professional critical care societies have proposed new clinical criteria that facilitate recognition of sepsis, the basic need for early detection and treatment remains unmet. It is in this context that the objective of this professional master's degree is to develop a machine learning-based model for the early and real-time prediction of sepsis from clinical data in intensive care units. This involves designing and implementing a functional predictive model to automatically identify a patient's risk of sepsis and make a positive or negative prediction of sepsis for each time window in the patient's clinical record.

## Objectives:
 Sepsis is a major public health concern with significant
morbidity, mortality, and healthcare expenses. Early detection
and antibiotic treatment of sepsis improve outcomes. However,
although professional critical care societies have proposed new
clinical criteria that aid sepsis recognition, the fundamental need
for early detection and treatment remains unmet. In response,
researchers have proposed algorithms for early sepsis detection,
but directly comparing such methods has not been possible because
of different patient cohorts, clinical variables and sepsis criteria,
prediction tasks, evaluation metrics, and other differences.
To address these issues, the PhysioNet/Computing in Cardiology
Challenge 2019 facilitated the development of automated,
open-source algorithms for the early detection of sepsis from clinical
data.

To do this the Knowledge Discovery in Databases (KDD) process will be adopted. This process consists of 4 steps:

1. Selecting and extracting the multivariate time series data set from the database.
2. Pre-processed and cleaned up the time series into a tidy dataset by exploring the data, handling missing data, and removing noise / outliers.
3. Building a predictive model allowing to associate the time series of biomedical measurements with a real-time severity indicator (probability of mortality) by implementing several machine learning algorithms.
4. Evaluation and validation. The use of two databases provides an opportunity to validate the different predictive models that will be produced.
## Brief reviex of ML 
```mermaid
flowchart TD
    %% Section Supervisée
    subgraph SB["🔍 Apprentissage Supervisé"]
        direction TB
        S[Supervisé] --> R[Régression]
        R --> RMSE["<b>RMSE=</b><br/>$$\\sqrt{\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2}$$"]
        RMSE --> RMSE_Exp["<small>• Mesure l'écart quadratique moyen<br>• Même unité que la variable cible<br>• Sensible aux outliers</small>"]
        
        R --> R2["<b>R²</b><br/>$$=1-\\frac{\\sum(y_i-\\hat{y}_i)^2}{\\sum(y_i-\\bar{y})^2}$$"]
        R2 --> R2_Exp["<small>• Proportion de variance expliquée<br>• Plage : 0 (nul) à 1 (parfait)<br>• Peut être négatif</small>"]
        
        R --> MAE["<b>MAE</b><br/>$$=\\frac{1}{n}\\sum_{i=1}^n|\\hat{y}_i-y_i|$$"]
        MAE --> MAE_Exp["<small>• Erreur absolue moyenne<br>• Moins sensible aux outliers<br>• Interprétation intuitive</small>"]
        
        S --> C[Classification]
        C --> Acc["<b>Accuracy</b><br/>$$=\\frac{TP+TN}{TP+TN+FP+FN}$$"]
        Acc --> Acc_Exp["<small>• Pourcentage correct<br>• Problème si classes déséquilibrées<br>• Bonne métrique de base</small>"]
        
        C --> Prec["<b>Précision</b><br/>$$=\\frac{TP}{TP+FP}$$"]
        Prec --> Prec_Exp["<small>• Qualité des prédictions positives<br>• Important quand FP coûtent cher<br>• Ex: diagnostic médical</small>"]
        
        C --> F1["<b>F1-Score</b><br/>$$=2\\times\\frac{Precision\\times Recall}{Precision+Recall}$$"]
        F1 --> F1_Exp["<small>• Moyenne harmonique<br>• Bon pour classes déséquilibrées<br>• Plage : 0 à 1</small>"]
        C --> Recall["<b>Rappel</b><br/>$$=\\frac{TP}{TP+FN}$$"]
        Recall --> Recall_Exp["<small>• Sensibilité<br>• Important quand FN sont critiques<br>• Ex : détection de fraude</small>"]
    
        Prec --> F1
        Recall --> F1
    end

    %% Section Non-Supervisée
    subgraph NSB["🌌 Apprentissage Non Supervise"]
        direction TB
        U[Non-Supervisé] --> Cl[Clustering]
        Cl --> Sil["<b>Silhouette</b><br/>$$=\\frac{b-a}{\\max(a,b)}$$"]
        Sil --> Sil_Exp["<small>• a = distance intra-cluster<br>• b = distance inter-cluster<br>• Valeur idéale ≈1</small>"]
        
        Cl --> DB["<b>Davies-Bouldin</b><br/>$$=\\frac{1}{k}\\sum_{i=1}^k\\max_{j\\neq i}\\left(\\frac{\\sigma_i+\\sigma_j}{d(c_i,c_j)}\\right)$$"]
        DB --> DB_Exp["<small>• σ = dispersion intra-cluster<br>• d = distance centroïdes<br>• Plus petit = mieux</small>"]
        
        U --> Rd[Réduction de dimension]
        Rd --> VE["<b>Variance Expliquée</b><br/>$$=\\sum_{i=1}^k\\lambda_i$$"]
        VE --> VE_Exp["<small>• % information conservée<br>• Règle : >95% idéal<br>• Cumulatif par composante</small>"]
    end

    %% Liaison verticale
    SB --> NSB

    %% Styles
    style SB fill:#f0f7ff,stroke:#0369a1
    style NSB fill:#f0fdf4,stroke:#047857
    

```
## The Data
Click [here](https://archive.physionet.org/users/shared/challenge-2019/) to download the complete training database (42 MB), consisting of two parts: training set A (20,336 subjects) and B (20,000 subjects).
Each training data file provides a table with measurements over time. Each column of the table provides a sequence of measurements over time (e.g., heart rate over several hours), where the header of the column describes the measurement. Each row of the table provides a collection of measurements at the same time (e.g., heart rate and oxygen level at the same time). The table is formatted in the following way:

HR |O2Sat|Temp|...|HospAdmTime|ICULOS|SepsisLabel
---|---|---|---|---|---|---|
NaN|  NaN| NaN|...|        -50|     1|          0
 86|   98| NaN|...|        -50|     2|          0
 75|  NaN| NaN|...|        -50|     3|          1
 99|  100|35.5|...|        -50|     4|          1
 
 
There are 40 time-dependent variables HR, O2Sat, Temp ..., HospAdmTime, which are described here. The final column, SepsisLabel, indicates the onset of sepsis according to the Sepsis-3 definition, where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number) indicate that there was no recorded measurement of a variable at the time interval.

Note: spaces were added to this example to improve readability. They will not be present in the data files.

***TABLE 1. Feature Summary***

Row|Measurement|Description
---|:---|:---
1| **HR**| Heart rate (beats/min)
2| **O2Sat**| Pulse oximetry (%)
3| **Temp** |Temperature (°C)
4|**SBP**| Systolic BP (mm Hg)
5| **MAP**| Mean arterial pressure (mm Hg)
6| **DBP**| Diastolic BP (mm Hg)
7| **Resp**| Respiration rate (breaths/min)
8| **Etco2**| End tidal carbon dioxide (mm Hg)
9| **BaseExcess**| Excess bicarbonate (mmol/L)
10| **Hco3**| Bicarbonate (mmol/L)
11 |**Fio2**| Fraction of inspired oxygen (%)
12 |**pH**| pH
13 |**Paco2**|Partial pressure of carbon dioxide fromarterial blood (mm Hg)
14 |**Sao2**| Oxygen saturation from arterial blood (%)
15 |**AST**| Aspartate transaminase (IU/L)
16 |**BUN**| Blood urea nitrogen (mg/dL)
17 |**Alkalinephos**| Alkaline phosphatase (IU/L)
18 |**Calcium**| Calcium (mg/dL)
19 |**Chloride**| Chloride (mmol/L)
20 |**Creatinine**| Creatinine (mg/dL)
21 |**Bilirubin**| direct Direct bilirubin (mg/dL)
22 |**Glucose**| Serum glucose (mg/dL)
23 |**Lactate**| Lactic acid (mg/dL)
24 |**Magnesium**| Magnesium (mmol/dL)
25 |**Phosphate**| Phosphate (mg/dL)
26 |**Potassium**| Potassium (mmol/L)
27 |**Bilirubin total**| Total bilirubin (mg/dL)
28 |**TroponinI**| Troponin I (ng/mL)
29 |**Hct**| Hematocrit (%)
30 |**Hgb**| Hemoglobin (g/dL)
31 |**PTT**| Partial thromboplastin time (s)
32 |**WBC**| Leukocyte count (count/L)
33 |**Fibrinogen**| Fibrinogen concentration (mg/dL)
34 |**Platelets**| Platelet count (count/mL)
35 |**Age**| Age (yr)
36 |**Gender**| Female (0) or male (1)
37 |**Unit 1**| Administrative identifier for ICU unit (medical ICU); false (0) or true (1)
38 |**Unit 2**| Administrative identifier for ICU unit (surgical ICU); false (0) or true (1)
39 |**HospAdmTime**| Time between hospital and ICU admission (hours since ICU admission)
40 |**ICULOS**| ICU length of stay (hours since ICU admission)
41 |**SepsisLabel**| For septic patients, SepsisLabel is 1 if $t \geq t_{sepsis}-6$ and $0$ if $t < t_{sepsis}-6$.For nonseptic patients, SepsisLabel is 0.

**Clinical time series data provided :** 
>* Vital signs (rows 1–8),
>* Laboratory values (rows 9–34)
>* Demographics (rows 35–40)
>* Outcome row 41.

Try this Binder for EDA : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nevermind78/capstone/main?filepath=EDA.ipynb)
