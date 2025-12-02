
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def identify_column_types(df, target_col):

    #exclude target column
    feature_cols = [col for col in df.columns if col != target_col]
    
    #Identify categorical columns (object type or low cardinality numeric)
    categorical_cols = []
    numeric_cols = []
    date_cols = []
    
    for col in feature_cols:
        #check if it's a date/datetime column
        if df[col].dtype == 'object':
            #try to parse as datetime
            try:
                #try parsing with different formats
                test_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if test_val:
                    pd.to_datetime(test_val, dayfirst=True, errors='raise')
                    date_cols.append(col)
            except:
                #Check cardinality drop if over 100 unique vlaues
                unique_count = df[col].nunique()
                if unique_count > 100:  # probably a date string
                    print(f"Dropping high-cardinality column '{col}' ({unique_count} unique values)")
                    continue
                categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            #check if its actually categorical (low unique values)
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05 and df[col].nunique() < 20:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            numeric_cols.append(col)
    
    return categorical_cols, numeric_cols, date_cols


def preprocess_data(df, target_col, test_size=0.2, random_state=42):

    #for energy efficiency dataset, making sure that heating and cooling load is dropped
    if target_col == "Heating_Load" and "Cooling_Load" in df.columns:
        df = df.drop(columns=["Cooling_Load"])
    elif target_col == "Cooling_Load" and "Heating_Load" in df.columns:
        df = df.drop(columns=["Heating_Load"])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    #date columns and convert to numeric features
    categorical_cols, numeric_cols, date_cols = identify_column_types(X, target_col)
    
    #Process date columns
    for col in date_cols:
        try:
            #European date format (dd-mm-yyyy)
            X[col] = pd.to_datetime(X[col], dayfirst=True, errors='coerce')
            #check if parsing worked
            if X[col].isna().all():
                #Try without dayfirst
                X[col] = pd.to_datetime(X[col], errors='coerce')
            
            if X[col].isna().all():
                raise ValueError("Could not parse any dates")
            
            #get the year, month, and day, day of week
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            #add the hour if it changes
            if hasattr(X[col].dt, 'hour'):
                hour_values = X[col].dt.hour
                if hour_values.nunique() > 1:
                    X[f'{col}_hour'] = hour_values
                    numeric_cols.append(f'{col}_hour')
            #fill NaN values with average
            for date_col in [f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek']:
                X[date_col] = X[date_col].fillna(X[date_col].median())
            #Drop original date column
            X = X.drop(columns=[col])
            #Add new date features to numeric columns
            numeric_cols.extend([f'{col}_year', f'{col}_month', f'{col}_day', 
                                f'{col}_dayofweek'])
            print(f"Converted date column '{col}' to numeric features")
        except Exception as e:
            print(f"Could not parse date column '{col}': {e}. Dropping it.")
            if col in X.columns:
                X = X.drop(columns=[col])
    
    #fix missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(X.mode().iloc[0] if len(X.mode()) > 0 else 'missing')
    
    #re identify column types after date processing (date-derived columns will be numeric)
    categorical_cols, numeric_cols, _ = identify_column_types(X, target_col)
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {len(numeric_cols)} numeric features")
    
    #Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    #split data using train_test_split with test size 0.2 for 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    #Fit and transform
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    print("Transforming test set...")
    X_test_processed = preprocessor.transform(X_test)
    print("Preprocessing complete!")
    
    #convert to DataFrame for better handling
    #get feature names after preprocessing
    feature_names = numeric_cols.copy()
    if categorical_cols:
        ohe = preprocessor.named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    print(f"Preprocessed training set shape: {X_train_df.shape}")
    print(f"Preprocessed test set shape: {X_test_df.shape}")
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor

