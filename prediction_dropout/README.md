Student Dropout and Academic Success Prediction


The purpose of this project is to develop a machine learning solution that can predict which students are at risk of academic failure or dropout at an early stage in higher education. By analyzing students’ enrollment information, demographic details, socio-economic factors, and academic performance in the first and second semesters, the system identifies students who may need additional support. 

Early identification allows universities and institutions to implement targeted interventions, such as academic counseling, tutoring, and mentorship programs, ultimately aiming to reduce dropout rates and improve student success.


The dataset used in this project was acquired from a higher education institution and combines information from multiple disjoint databases. It includes the following features:

- Previous qualification (grade) – Represents the grade obtained in the last completed qualification before enrollment.
- Admission grade – The grade achieved at admission.
- Age at enrollment – Age of the student at the time of enrollment.
- Curricular units 1st sem (approved) – Number of approved units in the first semester.
- Curricular units 1st sem (grade) – Average grade in first semester units.
- Curricular units 2nd sem (approved) – Number of approved units in the second semester.
- Curricular units 2nd sem (grade) – Average grade in second semester units.

 Project Structure
The project includes the following main components:
Data Gathering and Cleaningng**  
   - Handling missing values  
   - Detecting and removing outliers  
   - Descriptive statistical analysis  
Data Preprocessingng**  
   - Feature extraction and selection  
   - Feature tuning and encoding  
   - Train-test split  
Model Training and Tuningng**  
   - Training using CatBoost classifier  
   - Hyperparameter tuning  
   - Evaluation using accuracy, confusion matrix, and classification report  
Prediction Dashboardrd**  
   - Streamlit web application to allow user input  
   - Predict student dropout or graduation  
   - Recommend support for at-risk students  

Key Features
- Early prediction of students at risk of dropout  
- Recommendations for support interventions  
- Interactive dashboard for real-time predictions  

