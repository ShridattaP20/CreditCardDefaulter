import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRFClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    
    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBRFClassifier(),
                "KNeighbors": KNeighborsClassifier()
            }

            param={
             "Logistic Regression":{}, 
             
            "Decision Tree": {'criterion':["gini", "entropy", "log_loss"]},

            "Random Forest":{'n_estimators': [8,16,32,64,128,256]},

            "KNeighbors":{'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']},

            "XGBClassifier":{'learning_rate':[.1,.01,.05,.001],'n_estimators': [8,16,32,64,128,256]}
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=param)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model)

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            return acc_score
            




        except Exception as e:
            raise CustomException(e,sys)