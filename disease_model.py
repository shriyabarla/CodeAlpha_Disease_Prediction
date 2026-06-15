import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_universal_model(csv_path, target_column):
    print(f"--- Starting Medical ML Pipeline for: {csv_path} ---")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find '{csv_path}'. Ensure it is in the exact same folder as this script.")
        return

    print("Cleaning raw data (dropping IDs, standardizing target)...")
    
    if 'id' in df.columns: 
        df = df.drop('id', axis=1)
    if 'dataset' in df.columns: 
        df = df.drop('dataset', axis=1)
        
    df[target_column] = (df[target_column] > 0).astype(int)
        
    df = df.dropna(subset=[target_column])
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model (applying dynamic preprocessing)...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    roc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {roc:.4f}\n")
        
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    artifact_name = f"{csv_path.split('.')[0]}_pipeline.pkl"
    joblib.dump(pipeline, artifact_name)
    print(f"\nComplete pipeline saved to disk as '{artifact_name}'.")


DATASET_FILE = "heart_disease_uci.csv" 
TARGET_VARIABLE = "num" 

train_universal_model(DATASET_FILE, TARGET_VARIABLE)